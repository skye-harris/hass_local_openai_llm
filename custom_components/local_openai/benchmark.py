"""
Benchmark API for running voice requests against conversation agents.

Each conversation agent can be benchmarked through an admin-only endpoint at
``/api/local_openai/benchmark/{subentry_id}``. A request runs a single turn
through the regular conversation path, optionally as a voice satellite in an
area, so the agent sees the same prompt and tools as it would for a real
request. Tools annotated as read only run for real, every other tool call is
recorded and answered with a synthetic success, so devices are never controlled.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.components.http import KEY_HASS, HomeAssistantView, require_admin
from homeassistant.components.http.data_validator import RequestDataValidator
from homeassistant.components.intent import async_register_timer_handler
from homeassistant.const import CONF_MODEL
from homeassistant.core import Context, HomeAssistant, callback
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import intent, llm
from homeassistant.helpers.chat_session import CONVERSATION_TIMEOUT
from homeassistant.util.hass_dict import HassKey

from .const import DOMAIN

if TYPE_CHECKING:
    from collections.abc import Generator

    from aiohttp import web
    from homeassistant.components.intent import TimerEventType, TimerInfo
    from homeassistant.config_entries import ConfigEntry, ConfigSubentry
    from homeassistant.util.json import JsonObjectType, JsonValueType

    from .entity import GenerationMetadata

BENCHMARK_URL = "/api/local_openai/benchmark"

SATELLITE_IDENTIFIER_PREFIX = "benchmark_satellite"

DATA_BENCHMARK: HassKey[BenchmarkData] = HassKey(f"{DOMAIN}_benchmark")

REQUEST_SCHEMA = vol.Schema(
    {
        vol.Required("text"): cv.string,
        vol.Optional("area"): cv.string,
        vol.Optional("conversation_id"): cv.string,
        vol.Optional("language"): cv.string,
        # Replaces the instructions configured on the agent
        vol.Optional("prompt"): cv.string,
        # Any JSON value, tools may return plain text rather than an object
        vol.Optional("tool_results", default=dict): {cv.string: object},
    },
)

# Set for the duration of a benchmark turn. The tool call tasks created by the chat
# log copy the context, so the wrapped tools see the run of the turn that called them.
_current_run: ContextVar[BenchmarkRun | None] = ContextVar(
    f"{DOMAIN}_benchmark_run",
    default=None,
)


@dataclass
class BenchmarkRun:
    """Results collected during a single benchmark turn."""

    prompt: str | None = None
    tool_results: dict[str, JsonValueType] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    iterations: list[dict[str, Any]] = field(default_factory=list)

    def get_mock(self, tool_name: str) -> tuple[bool, JsonValueType]:
        """
        Return whether a tool is mocked and its result.

        Matched with or without the tool namespace. A mocked result may be null.
        """
        for key in (tool_name, tool_name.rpartition("__")[2]):
            if key in self.tool_results:
                return True, self.tool_results[key]
        return False, None


@dataclass
class BenchmarkConversation:
    """A conversation started by the benchmark API, which may be continued."""

    subentry_id: str
    prompt: str | None
    last_used: float


@dataclass
class BenchmarkData:
    """Benchmark state shared across requests."""

    # Conversation ID -> conversation, only these may be continued
    conversations: dict[str, BenchmarkConversation] = field(default_factory=dict)
    # Config entry ID -> lock, so runs against one server don't skew each other's timings
    locks: dict[str, asyncio.Lock] = field(default_factory=dict)

    @callback
    def async_prune_conversations(self) -> None:
        """Forget conversations whose chat session has expired."""
        expired = time.monotonic() - CONVERSATION_TIMEOUT.total_seconds()
        self.conversations = {
            conversation_id: value
            for conversation_id, value in self.conversations.items()
            if value.last_used > expired
        }


@callback
def _get_data(hass: HomeAssistant) -> BenchmarkData:
    """Return the shared benchmark state."""
    return hass.data.setdefault(DATA_BENCHMARK, BenchmarkData())


@callback
def async_register_views(hass: HomeAssistant) -> None:
    """Register the benchmark API views."""
    _get_data(hass)
    hass.http.register_view(BenchmarkAgentsView())
    hass.http.register_view(BenchmarkAgentView())


class BenchmarkTool(llm.Tool):
    """Wraps a tool so that a benchmark only runs it when it is read only."""

    def __init__(self, tool: llm.Tool) -> None:
        """Initialize the wrapper, keeping the tool definition the model sees."""
        self.tool = tool
        self.name = tool.name
        self.title = tool.title
        self.description = tool.description
        self.parameters = tool.parameters
        self.annotations = tool.annotations
        self.integration = tool.integration

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> llm.ToolResult:
        """Run a read only tool, otherwise record the call and report success."""
        run = _current_run.get()
        record: dict[str, Any] = {
            "iteration": len(run.iterations) + 1 if run else None,
            "name": self.name,
            "arguments": tool_input.tool_args,
            "executed": False,
            "mocked": False,
        }
        if run is not None:
            run.tool_calls.append(record)

        mocked, mock = run.get_mock(self.name) if run is not None else (False, None)
        try:
            if mocked:
                record["mocked"] = True
                # Wrapped like a tool that returns plain JSON, so the model sees the
                # same tool message as it would from the real tool
                result = llm.ToolResult(data=mock)  # type: ignore[arg-type]
            elif self.annotations.read_only:
                record["executed"] = True
                result = await self.tool.async_call(hass, tool_input, llm_context)
                if not isinstance(result, llm.ToolResult):
                    result = llm.ToolResult(data=result)
            else:
                # Invalid arguments fail the same way they would if the tool ran
                self.tool.parameters(tool_input.tool_args)
                result = llm.ToolResult(data=_action_result(self.tool, llm_context))
        except Exception as err:
            record["error"] = True
            record["result"] = {"error": type(err).__name__, "error_text": str(err)}
            raise

        record["error"] = result.error
        record["result"] = result.data
        return result


def _action_result(tool: llm.Tool, llm_context: llm.LLMContext) -> JsonObjectType:
    """Return a successful result for a tool that was not executed."""
    while isinstance(tool, llm.NamespacedTool):
        tool = tool.tool
    if isinstance(tool, llm.IntentTool):
        # Same shape as a real intent response, without the targets it would resolve
        return llm.IntentResponseDict(
            intent.IntentResponse(language=llm_context.language or ""),
        )
    return {"success": True}


@callback
def async_wrap_tools(chat_log: conversation.ChatLog) -> None:
    """Wrap the tools of a benchmark turn so that actions are not executed."""
    if _current_run.get() is None or chat_log.llm_api is None:
        return

    tools = chat_log.llm_api.tools
    if all(isinstance(tool, BenchmarkTool) for tool in tools):
        return

    # Replace the API instance rather than its tools, so nothing else that holds the
    # instance is affected. The chat log keeps it for the following turns.
    chat_log.llm_api = dataclasses.replace(
        chat_log.llm_api,
        tools=[
            tool if isinstance(tool, BenchmarkTool) else BenchmarkTool(tool)
            for tool in tools
        ],
    )


@callback
def async_get_prompt() -> str | None:
    """Return the prompt replacing the agent instructions for a benchmark turn."""
    if (run := _current_run.get()) is None:
        return None
    return run.prompt


@callback
def async_record_iteration(
    iteration: int,
    model: str,
    content: str | None,
    meta: GenerationMetadata,
    timings: dict[str, Any],
) -> None:
    """Record the metadata of an LLM request made during a benchmark turn."""
    if (run := _current_run.get()) is None:
        return

    usage = meta.usage
    if usage is not None and hasattr(usage, "model_dump"):
        usage = usage.model_dump(exclude_none=True)

    run.iterations.append(
        {
            "iteration": iteration,
            "model": model,
            "content": content,
            "tools_called": meta.tool_names or [],
            "finish_reason": meta.finish_reason,
            "time_to_first_token_ms": meta.time_to_first_token,
            "duration_ms": round((time.time() - meta.request_start) * 1000, 2),
            "usage": usage,
            "timings": timings or None,
        },
    )


def _satellite_identifier(area_id: str) -> tuple[str, str]:
    """Return the device identifier of a benchmark satellite, unique per config entry."""
    return (DOMAIN, f"{SATELLITE_IDENTIFIER_PREFIX}_{area_id}")


def _ignore_timer_event(event_type: TimerEventType, timer: TimerInfo) -> None:
    """Ignore timer events, benchmark satellites only need to advertise timers."""


@contextmanager
def _async_satellite(
    hass: HomeAssistant,
    entry: ConfigEntry,
    area: ar.AreaEntry,
) -> Generator[str]:
    """
    Provide a virtual voice satellite device in the area for a single request.

    Home Assistant only learns the area of a request from a device in the device
    registry, so the device exists while the request runs and is removed after.
    """
    dev_reg = dr.async_get(hass)
    # Recreating a removed device restores its ID, so it is stable between turns
    device = dev_reg.async_get_or_create(
        config_entry_id=entry.entry_id,
        identifiers={_satellite_identifier(area.id)},
        name=f"Benchmark satellite ({area.name})",
        model="Benchmark satellite",
        entry_type=dr.DeviceEntryType.SERVICE,
    )
    if device.area_id != area.id:
        dev_reg.async_update_device(device.id, area_id=area.id)

    # Voice satellites handle timers, which exposes the timer tools to the agent
    unsub_timers = async_register_timer_handler(hass, device.id, _ignore_timer_event)
    try:
        yield device.id
    finally:
        unsub_timers()
        dev_reg.async_remove_device(device.id)


@callback
def async_remove_satellites(hass: HomeAssistant, entry_id: str) -> None:
    """Remove benchmark satellites left behind by a request that did not finish."""
    dev_reg = dr.async_get(hass)
    prefix = f"{SATELLITE_IDENTIFIER_PREFIX}_"
    for device in dr.async_entries_for_config_entry(dev_reg, entry_id):
        if any(
            domain == DOMAIN and identifier.startswith(prefix)
            for domain, identifier in device.identifiers
        ):
            dev_reg.async_remove_device(device.id)


@callback
def _async_get_agents(
    hass: HomeAssistant,
) -> list[tuple[ConfigEntry, ConfigSubentry, str]]:
    """Return the loaded conversation agents with their entity IDs."""
    ent_reg = er.async_get(hass)
    agents = []
    for entry in hass.config_entries.async_loaded_entries(DOMAIN):
        for subentry in entry.subentries.values():
            if subentry.subentry_type != "conversation":
                continue
            if entity_id := ent_reg.async_get_entity_id(
                conversation.DOMAIN,
                DOMAIN,
                subentry.subentry_id,
            ):
                agents.append((entry, subentry, entity_id))
    return agents


class BenchmarkAgentsView(HomeAssistantView):
    """List the conversation agents and areas that can be benchmarked."""

    url = BENCHMARK_URL
    name = "api:local_openai:benchmark"

    @require_admin
    async def get(self, request: web.Request) -> web.Response:
        """Return the benchmark endpoints of every conversation agent."""
        hass = request.app[KEY_HASS]
        agents = [
            {
                "agent_id": entity_id,
                "name": subentry.title,
                "model": subentry.data.get(CONF_MODEL),
                "server": entry.title,
                "url": f"{BENCHMARK_URL}/{subentry.subentry_id}",
            }
            for entry, subentry, entity_id in _async_get_agents(hass)
        ]
        areas = [
            {"area_id": area.id, "name": area.name, "floor_id": area.floor_id}
            for area in ar.async_get(hass).async_list_areas()
        ]
        return self.json({"agents": agents, "areas": areas})


class BenchmarkAgentView(HomeAssistantView):
    """Run a benchmark turn against a single conversation agent."""

    url = BENCHMARK_URL + "/{subentry_id}"
    name = "api:local_openai:benchmark:agent"

    @require_admin
    @RequestDataValidator(REQUEST_SCHEMA)
    async def post(
        self,
        request: web.Request,
        data: dict[str, Any],
        subentry_id: str,
    ) -> web.Response:
        """Run one conversation turn without executing any actions."""
        hass = request.app[KEY_HASS]
        benchmark = _get_data(hass)

        agent = next(
            (
                agent
                for agent in _async_get_agents(hass)
                if agent[1].subentry_id == subentry_id
            ),
            None,
        )
        if agent is None or conversation.async_get_agent(hass, agent[2]) is None:
            return self.json_message(
                "Conversation agent not found",
                HTTPStatus.NOT_FOUND,
            )
        entry, _, entity_id = agent

        # Only conversations started here are continued, a real conversation would
        # otherwise keep the wrapped tools for its next turns.
        benchmark.async_prune_conversations()
        conversation_id = data.get("conversation_id")
        prompt = data.get("prompt")
        if conversation_id is not None:
            existing = benchmark.conversations.get(conversation_id)
            if existing is None or existing.subentry_id != subentry_id:
                return self.json_message(
                    "Unknown conversation_id, only conversations started by this benchmark endpoint can be continued",
                    HTTPStatus.BAD_REQUEST,
                )
            # Home Assistant keeps the system prompt of a conversation between turns
            # when the agent has an LLM API, so the prompt is set by the first turn
            if prompt is None:
                prompt = existing.prompt
            elif prompt != existing.prompt:
                return self.json_message(
                    "The prompt cannot change within a conversation",
                    HTTPStatus.BAD_REQUEST,
                )

        area: ar.AreaEntry | None = None
        if (area_ref := data.get("area")) is not None:
            area_reg = ar.async_get(hass)
            area = area_reg.async_get_area(area_ref) or area_reg.async_get_area_by_name(
                area_ref,
            )
            if area is None:
                return self.json_message(
                    f"Unknown area: {area_ref}",
                    HTTPStatus.BAD_REQUEST,
                )
            # Voice satellites converse without a user
            context = Context()
        else:
            context = self.context(request)

        run = BenchmarkRun(prompt=prompt, tool_results=data["tool_results"])
        lock = benchmark.locks.setdefault(entry.entry_id, asyncio.Lock())
        async with lock:
            # Inside the lock, so a request never removes the satellite of another
            with (
                _async_satellite(hass, entry, area)
                if area is not None
                else nullcontext()
            ) as device_id:
                token = _current_run.set(run)
                start = time.perf_counter()
                try:
                    # Agent errors are returned as an error response rather than raised
                    result = await conversation.async_converse(
                        hass,
                        text=data["text"],
                        conversation_id=conversation_id,
                        context=context,
                        language=data.get("language") or hass.config.language,
                        agent_id=entity_id,
                        device_id=device_id,
                    )
                finally:
                    _current_run.reset(token)
                duration_ms = round((time.perf_counter() - start) * 1000, 2)

        if result.conversation_id:
            benchmark.conversations[result.conversation_id] = BenchmarkConversation(
                subentry_id=subentry_id,
                prompt=prompt,
                last_used=time.monotonic(),
            )

        response = result.response
        return self.json(
            {
                "conversation_id": result.conversation_id,
                "agent_id": entity_id,
                "area_id": area.id if area else None,
                "speech": response.speech.get("plain", {}).get("speech", ""),
                "response_type": response.response_type.value,
                "error_code": response.error_code.value
                if response.error_code
                else None,
                "continue_conversation": result.continue_conversation,
                "tool_calls": run.tool_calls,
                "iterations": run.iterations,
                "duration_ms": duration_ms,
            },
        )
