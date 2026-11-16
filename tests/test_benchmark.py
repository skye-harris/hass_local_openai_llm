"""Tests for the benchmark API."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from http import HTTPStatus
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import probatio
import pytest
from custom_components.local_openai.benchmark import (
    BENCHMARK_URL,
    BenchmarkRun,
    BenchmarkTool,
    _current_run,
    async_wrap_tools,
)
from custom_components.local_openai.const import (
    CONF_ALWAYS_CONTINUE_CONVERSATION,
    CONF_BASE_URL,
    CONF_SERVER_TYPE,
    DOMAIN,
    SERVER_TYPE_GENERIC,
    TOOL_NAME_STOP_TALKING,
)
from custom_components.local_openai.conversation import LocalAiConversationEntity
from custom_components.local_openai.tools.stop_talking import StopTalkingTool
from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import (
    async_expose_entity,
)
from homeassistant.components.intent import async_device_supports_timers
from homeassistant.config_entries import ConfigSubentryData
from homeassistant.const import CONF_LLM_HASS_API, CONF_MODEL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import llm
from homeassistant.setup import async_setup_component
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from pytest_homeassistant.common import MockConfigEntry, async_mock_service
from pytest_homeassistant.typing import ClientSessionGenerator

SUBENTRY_ID = "benchmark_agent"
TURN_ON = "intent__HassTurnOn"
LIVE_CONTEXT = "homeassistant__GetLiveContext"


class FakeTool(llm.Tool):
    """Tool that records whether it ran."""

    def __init__(self, name: str, *, read_only: bool) -> None:
        """Initialize the tool."""
        self.name = name
        self.description = f"{name} description"
        self.parameters = probatio.Schema({probatio.Required("target"): str})
        self.annotations = llm.ToolAnnotations(read_only=read_only)
        self.integration = "test"
        self.calls: list[dict[str, Any]] = []

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> llm.ToolResult:
        """Record the call."""
        self.calls.append(tool_input.tool_args)
        return llm.ToolResult(data={"ran": True})


def _llm_context() -> llm.LLMContext:
    return llm.LLMContext(
        platform=DOMAIN,
        context=None,
        language="en",
        assistant=conversation.DOMAIN,
        device_id=None,
    )


async def _call(tool: llm.Tool, args: dict[str, Any]) -> llm.ToolResult:
    return await tool.async_call(
        MagicMock(),
        llm.ToolInput(tool_name=tool.name, tool_args=args),
        _llm_context(),
    )


class TestBenchmarkTool:
    """Test which tool calls run during a benchmark."""

    async def test_read_only_tool_runs(self) -> None:
        """Test that read only tools are executed and recorded."""
        inner = FakeTool("read", read_only=True)
        run = BenchmarkRun()
        token = _current_run.set(run)
        try:
            result = await _call(BenchmarkTool(inner), {"target": "x"})
        finally:
            _current_run.reset(token)

        assert inner.calls == [{"target": "x"}]
        assert result.data == {"ran": True}
        assert run.tool_calls == [
            {
                "iteration": 1,
                "name": "read",
                "arguments": {"target": "x"},
                "executed": True,
                "mocked": False,
                "error": False,
                "result": {"ran": True},
            },
        ]

    async def test_action_tool_is_not_run(self) -> None:
        """Test that tools that are not read only are recorded but not executed."""
        inner = FakeTool("write", read_only=False)
        run = BenchmarkRun()
        token = _current_run.set(run)
        try:
            result = await _call(BenchmarkTool(inner), {"target": "x"})
        finally:
            _current_run.reset(token)

        assert inner.calls == []
        assert result.data == {"success": True}
        assert run.tool_calls[0]["executed"] is False
        assert run.tool_calls[0]["result"] == {"success": True}

    async def test_action_tool_is_not_run_without_run(self) -> None:
        """Test that wrapped tools still never run actions outside a benchmark turn."""
        inner = FakeTool("write", read_only=False)

        result = await _call(BenchmarkTool(inner), {"target": "x"})

        assert inner.calls == []
        assert result.data == {"success": True}

    async def test_action_tool_invalid_arguments(self) -> None:
        """Test that invalid arguments fail like they would for a real call."""
        inner = FakeTool("write", read_only=False)
        run = BenchmarkRun()
        token = _current_run.set(run)
        try:
            with pytest.raises(probatio.Invalid):
                await _call(BenchmarkTool(inner), {"wrong": "x"})
        finally:
            _current_run.reset(token)

        assert inner.calls == []
        assert run.tool_calls[0]["error"] is True
        assert run.tool_calls[0]["result"]["error"] == "MultipleInvalid"

    @pytest.mark.parametrize("key", ["test__read", "read"])
    @pytest.mark.parametrize("mock", [{"mocked": True}, "Sunny, high of 22", None])
    async def test_mocked_result(self, key: str, mock: Any) -> None:
        """Test that mocked results of any JSON type replace a tool, with or without its namespace."""
        inner = FakeTool("test__read", read_only=True)
        run = BenchmarkRun(tool_results={key: mock})
        token = _current_run.set(run)
        try:
            result = await _call(BenchmarkTool(inner), {"target": "x"})
        finally:
            _current_run.reset(token)

        assert inner.calls == []
        assert result.data == mock
        assert run.tool_calls[0]["mocked"] is True
        assert run.tool_calls[0]["result"] == mock


class TestWrapTools:
    """Test wrapping the tools of a chat log."""

    def _chat_log(self, tools: list[llm.Tool]) -> MagicMock:
        chat_log = MagicMock(spec=conversation.ChatLog)
        chat_log.conversation_id = "test_conversation_id"
        chat_log.llm_api = llm.APIInstance(
            api=MagicMock(),
            api_prompt="",
            llm_context=_llm_context(),
            tools=tools,
        )
        return chat_log

    def test_not_wrapped_outside_benchmark(self) -> None:
        """Test that regular turns keep their tools."""
        tool = FakeTool("write", read_only=False)
        chat_log = self._chat_log([tool])
        api = chat_log.llm_api

        async_wrap_tools(chat_log)

        assert chat_log.llm_api is api
        assert api.tools == [tool]

    def test_wrapped_once(self) -> None:
        """Test that the tools are wrapped once, keeping their definition."""
        tool = FakeTool("write", read_only=False)
        chat_log = self._chat_log([tool])
        api = chat_log.llm_api

        token = _current_run.set(BenchmarkRun())
        try:
            async_wrap_tools(chat_log)
            wrapped_api = chat_log.llm_api
            async_wrap_tools(chat_log)
        finally:
            _current_run.reset(token)

        # The original instance is left untouched
        assert api.tools == [tool]
        assert chat_log.llm_api is wrapped_api
        (wrapped,) = wrapped_api.tools
        assert isinstance(wrapped, BenchmarkTool)
        assert wrapped.tool is tool
        assert wrapped.name == tool.name
        assert wrapped.description == tool.description
        assert wrapped.parameters is tool.parameters
        assert wrapped.annotations is tool.annotations

    async def test_stop_talking_not_duplicated(self, hass: HomeAssistant) -> None:
        """Test that a wrapped stop_talking tool is not injected again on the next turn."""
        entry = MagicMock()
        entry.data = {}
        subentry = MagicMock()
        subentry.data = {
            CONF_MODEL: "test-model",
            CONF_LLM_HASS_API: ["assist"],
            CONF_ALWAYS_CONTINUE_CONVERSATION: True,
        }
        entity = LocalAiConversationEntity(entry, subentry)
        entity.hass = hass
        entity._async_handle_chat_log = AsyncMock(return_value=False)

        chat_log = self._chat_log([BenchmarkTool(StopTalkingTool())])
        chat_log.content = [conversation.AssistantContent(agent_id="a", content="")]
        chat_log.continue_conversation = False
        chat_log.async_provide_llm_data = AsyncMock()

        api = MagicMock()
        api.id = "assist"
        with patch(
            "custom_components.local_openai.conversation.llm.async_get_apis",
            return_value=[api],
        ):
            await entity._async_handle_message(
                MagicMock(spec=conversation.ConversationInput),
                chat_log,
            )

        assert [tool.name for tool in chat_log.llm_api.tools] == [
            TOOL_NAME_STOP_TALKING,
        ]


def _chunk(delta: ChoiceDelta, finish_reason: str | None = None) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="test",
        created=0,
        model="test-model",
        object="chat.completion.chunk",
        choices=[Choice(index=0, delta=delta, finish_reason=finish_reason)],
    )


async def _stream(*chunks: ChatCompletionChunk) -> AsyncGenerator[ChatCompletionChunk]:
    for chunk in chunks:
        yield chunk


def _tool_call_stream(
    *calls: tuple[str, dict[str, Any]],
) -> AsyncGenerator[ChatCompletionChunk]:
    tool_calls = [
        ChoiceDeltaToolCall(
            index=index,
            id=f"call_{index}",
            type="function",
            function=ChoiceDeltaToolCallFunction(name=name, arguments=json.dumps(args)),
        )
        for index, (name, args) in enumerate(calls)
    ]
    return _stream(
        _chunk(ChoiceDelta(role="assistant", tool_calls=tool_calls)),
        _chunk(ChoiceDelta(), finish_reason="tool_calls"),
    )


def _text_stream(text: str) -> AsyncGenerator[ChatCompletionChunk]:
    return _stream(
        _chunk(ChoiceDelta(role="assistant", content=text)),
        _chunk(ChoiceDelta(), finish_reason="stop"),
    )


@pytest.fixture
def socket_enabled() -> None:
    """Satisfy hass_client, sockets are not blocked as pytest-socket is disabled."""


@pytest.fixture
async def client_mock() -> MagicMock:
    """Return a mocked OpenAI client."""
    client = MagicMock()
    client.with_options.return_value.models.list = MagicMock(
        __aiter__=MagicMock(
            return_value=AsyncMock(__anext__=AsyncMock(side_effect=StopAsyncIteration)),
        ),
    )
    client.chat.completions.create = AsyncMock()
    return client


@pytest.fixture
async def config_entry(
    hass: HomeAssistant,
    enable_custom_integrations: None,
    client_mock: MagicMock,
) -> MockConfigEntry:
    """Set up the integration with a conversation agent and a light in the kitchen."""
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "http", {})
    assert await async_setup_component(hass, "conversation", {})

    kitchen = ar.async_get(hass).async_create("Kitchen")
    ent_reg = er.async_get(hass)
    light = ent_reg.async_get_or_create(
        "light",
        "test",
        "kitchen_light",
        suggested_object_id="kitchen",
        original_name="Kitchen light",
    )
    ent_reg.async_update_entity(light.entity_id, area_id=kitchen.id)
    hass.states.async_set(light.entity_id, "off", {"friendly_name": "Kitchen light"})
    async_expose_entity(hass, conversation.DOMAIN, light.entity_id, True)

    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Test Server",
        version=5,
        data={
            CONF_BASE_URL: "http://localhost:8080/v1",
            CONF_SERVER_TYPE: SERVER_TYPE_GENERIC,
        },
        subentries_data=[
            ConfigSubentryData(
                data={CONF_MODEL: "test-model", CONF_LLM_HASS_API: ["assist"]},
                subentry_id=SUBENTRY_ID,
                subentry_type="conversation",
                title="Benchmark Agent",
                unique_id=None,
            ),
        ],
    )
    entry.add_to_hass(hass)
    with patch("custom_components.local_openai.AsyncOpenAI", return_value=client_mock):
        assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry


def _satellite(hass: HomeAssistant, entry: MockConfigEntry) -> dr.DeviceEntry | None:
    return dr.async_get(hass).async_get_device_by_identifier(
        (DOMAIN, "benchmark_satellite_kitchen"),
        entry.entry_id,
    )


def _record_satellite(
    hass: HomeAssistant,
    entry: MockConfigEntry,
    client_mock: MagicMock,
    streams: list[AsyncGenerator[ChatCompletionChunk]],
) -> list[dr.DeviceEntry | None]:
    """Return the satellite seen by each LLM request, as it only exists during one."""
    seen: list[dr.DeviceEntry | None] = []
    remaining = iter(streams)

    async def create(**kwargs: Any) -> AsyncGenerator[ChatCompletionChunk]:
        seen.append(_satellite(hass, entry))
        return next(remaining)

    client_mock.chat.completions.create.side_effect = create
    return seen


class TestBenchmarkView:
    """Test the benchmark HTTP API end to end."""

    async def test_satellite_turn_does_not_control_devices(
        self,
        hass: HomeAssistant,
        hass_client: ClientSessionGenerator,
        config_entry: MockConfigEntry,
        client_mock: MagicMock,
    ) -> None:
        """Test a satellite turn runs read only tools and skips actions."""
        turn_on_calls = async_mock_service(hass, "light", "turn_on")
        seen = _record_satellite(
            hass,
            config_entry,
            client_mock,
            [
                _tool_call_stream(
                    (LIVE_CONTEXT, {}),
                    (TURN_ON, {"area": "Kitchen", "domain": ["light"]}),
                ),
                _text_stream("Turned on the kitchen lights."),
            ],
        )

        client = await hass_client()
        resp = await client.post(
            f"{BENCHMARK_URL}/{SUBENTRY_ID}",
            json={"text": "turn on the lights", "area": "kitchen"},
        )
        assert resp.status == HTTPStatus.OK
        result = await resp.json()

        # Nothing was controlled
        assert turn_on_calls == []
        assert hass.states.get("light.kitchen").state == "off"

        # The satellite was in the area during the request, and removed after it
        satellite, _ = seen
        assert satellite is not None
        assert satellite.area_id == "kitchen"
        assert _satellite(hass, config_entry) is None
        assert not async_device_supports_timers(hass, satellite.id)

        assert result["speech"] == "Turned on the kitchen lights."
        assert result["response_type"] == "action_done"
        assert result["agent_id"] == "conversation.benchmark_agent"
        assert result["area_id"] == "kitchen"
        assert result["conversation_id"]

        live_context, turn_on = result["tool_calls"]
        assert live_context["name"] == LIVE_CONTEXT
        assert live_context["executed"] is True
        assert "Kitchen light" in json.dumps(live_context["result"])
        assert turn_on == {
            "iteration": 1,
            "name": TURN_ON,
            "arguments": {"area": "Kitchen", "domain": ["light"]},
            "executed": False,
            "mocked": False,
            "error": False,
            "result": {
                "speech": {},
                "response_type": "action_done",
                "data": {"success": [], "failed": []},
            },
        }

        assert [iteration["iteration"] for iteration in result["iterations"]] == [1, 2]
        assert result["iterations"][0]["tools_called"] == [LIVE_CONTEXT, TURN_ON]
        assert result["iterations"][1]["content"] == "Turned on the kitchen lights."

        # The agent was told where the satellite is, and given the timer tools
        request = client_mock.chat.completions.create.call_args_list[0].kwargs
        system_prompt = request["messages"][0]["content"]
        assert "You are in area Kitchen" in system_prompt
        tool_names = {tool["function"]["name"] for tool in request["tools"]}
        assert "intent__HassStartTimer" in tool_names

    async def test_turn_without_area(
        self,
        hass: HomeAssistant,
        hass_client: ClientSessionGenerator,
        config_entry: MockConfigEntry,
        client_mock: MagicMock,
    ) -> None:
        """Test a turn without an area is not a satellite."""
        client_mock.chat.completions.create.side_effect = [_text_stream("Hello")]

        client = await hass_client()
        resp = await client.post(f"{BENCHMARK_URL}/{SUBENTRY_ID}", json={"text": "hi"})
        assert resp.status == HTTPStatus.OK
        result = await resp.json()

        assert result["tool_calls"] == []
        assert _satellite(hass, config_entry) is None
        request = client_mock.chat.completions.create.call_args_list[0].kwargs
        assert (
            "This device is not able to start timers."
            in request["messages"][0]["content"]
        )

    @pytest.mark.parametrize(
        "mock",
        [{"error": "device offline"}, "The kitchen light is offline.", None],
    )
    async def test_mocked_tool_result(
        self,
        hass_client: ClientSessionGenerator,
        config_entry: MockConfigEntry,
        client_mock: MagicMock,
        mock: Any,
    ) -> None:
        """Test that the client can provide tool results of any JSON type."""
        client_mock.chat.completions.create.side_effect = [
            _tool_call_stream((TURN_ON, {"name": "Kitchen light"})),
            _text_stream("That light is broken."),
        ]

        client = await hass_client()
        resp = await client.post(
            f"{BENCHMARK_URL}/{SUBENTRY_ID}",
            json={
                "text": "turn on the kitchen light",
                "tool_results": {"HassTurnOn": mock},
            },
        )
        assert resp.status == HTTPStatus.OK
        result = await resp.json()

        (call,) = result["tool_calls"]
        assert call["mocked"] is True
        assert call["result"] == mock
        messages = client_mock.chat.completions.create.call_args_list[1].kwargs[
            "messages"
        ]
        # Serialized the same way as the result of a real tool
        (tool_message,) = [m for m in messages if m["role"] == "tool"]
        assert tool_message["content"] == json.dumps(mock)

    async def test_continue_conversation(
        self,
        hass_client: ClientSessionGenerator,
        config_entry: MockConfigEntry,
        client_mock: MagicMock,
    ) -> None:
        """Test that only conversations started by the benchmark can be continued."""
        client_mock.chat.completions.create.side_effect = [
            _text_stream("Which light?"),
            _text_stream("Done."),
        ]
        client = await hass_client()
        url = f"{BENCHMARK_URL}/{SUBENTRY_ID}"

        resp = await client.post(
            url, json={"text": "hi", "conversation_id": "01JUNKNOWN"}
        )
        assert resp.status == HTTPStatus.BAD_REQUEST

        first = await (
            await client.post(url, json={"text": "turn on the light"})
        ).json()
        resp = await client.post(
            url,
            json={"text": "the kitchen", "conversation_id": first["conversation_id"]},
        )
        assert resp.status == HTTPStatus.OK
        second = await resp.json()
        assert second["conversation_id"] == first["conversation_id"]
        assert second["speech"] == "Done."

    async def test_errors(
        self,
        hass_client: ClientSessionGenerator,
        hass_read_only_access_token: str,
        config_entry: MockConfigEntry,
    ) -> None:
        """Test the error responses."""
        client = await hass_client()
        resp = await client.post(f"{BENCHMARK_URL}/unknown", json={"text": "hi"})
        assert resp.status == HTTPStatus.NOT_FOUND

        resp = await client.post(
            f"{BENCHMARK_URL}/{SUBENTRY_ID}",
            json={"text": "hi", "area": "attic"},
        )
        assert resp.status == HTTPStatus.BAD_REQUEST

        resp = await client.post(f"{BENCHMARK_URL}/{SUBENTRY_ID}", json={})
        assert resp.status == HTTPStatus.BAD_REQUEST

        read_only_client = await hass_client(hass_read_only_access_token)
        resp = await read_only_client.post(
            f"{BENCHMARK_URL}/{SUBENTRY_ID}",
            json={"text": "hi"},
        )
        assert resp.status == HTTPStatus.UNAUTHORIZED
        resp = await read_only_client.get(BENCHMARK_URL)
        assert resp.status == HTTPStatus.UNAUTHORIZED

    async def test_list_agents(
        self,
        hass_client: ClientSessionGenerator,
        config_entry: MockConfigEntry,
    ) -> None:
        """Test listing the agents and areas that can be benchmarked."""
        client = await hass_client()
        resp = await client.get(BENCHMARK_URL)
        assert resp.status == HTTPStatus.OK
        assert await resp.json() == {
            "agents": [
                {
                    "agent_id": "conversation.benchmark_agent",
                    "name": "Benchmark Agent",
                    "model": "test-model",
                    "server": "Test Server",
                    "url": f"{BENCHMARK_URL}/{SUBENTRY_ID}",
                },
            ],
            "areas": [{"area_id": "kitchen", "name": "Kitchen", "floor_id": None}],
        }

    async def test_satellite_removed_after_failed_request(
        self,
        hass: HomeAssistant,
        hass_client: ClientSessionGenerator,
        config_entry: MockConfigEntry,
        client_mock: MagicMock,
    ) -> None:
        """Test that the satellite is removed when the LLM request fails."""
        client_mock.chat.completions.create.side_effect = openai.APIConnectionError(
            request=MagicMock(),
        )
        client = await hass_client()
        resp = await client.post(
            f"{BENCHMARK_URL}/{SUBENTRY_ID}",
            json={"text": "hi", "area": "Kitchen"},
        )
        assert resp.status == HTTPStatus.OK
        assert (await resp.json())["response_type"] == "error"
        assert _satellite(hass, config_entry) is None

    async def test_satellite_id_stable_between_requests(
        self,
        hass: HomeAssistant,
        hass_client: ClientSessionGenerator,
        config_entry: MockConfigEntry,
        client_mock: MagicMock,
    ) -> None:
        """Test that a recreated satellite keeps its ID, so follow ups match the first turn."""
        seen = _record_satellite(
            hass,
            config_entry,
            client_mock,
            [_text_stream("Which light?"), _text_stream("Done.")],
        )
        client = await hass_client()
        url = f"{BENCHMARK_URL}/{SUBENTRY_ID}"
        first = await (
            await client.post(
                url, json={"text": "turn on the light", "area": "Kitchen"}
            )
        ).json()
        await client.post(
            url,
            json={
                "text": "the ceiling one",
                "area": "Kitchen",
                "conversation_id": first["conversation_id"],
            },
        )

        first_satellite, second_satellite = seen
        assert first_satellite is not None
        assert second_satellite is not None
        assert first_satellite.id == second_satellite.id
        assert _satellite(hass, config_entry) is None

    async def test_leftover_satellites_removed_on_setup(
        self,
        hass: HomeAssistant,
        config_entry: MockConfigEntry,
        client_mock: MagicMock,
    ) -> None:
        """Test that a satellite left behind by a shutdown during a request is removed."""
        dr.async_get(hass).async_get_or_create(
            config_entry_id=config_entry.entry_id,
            identifiers={(DOMAIN, "benchmark_satellite_kitchen")},
        )

        with patch(
            "custom_components.local_openai.AsyncOpenAI",
            return_value=client_mock,
        ):
            assert await hass.config_entries.async_reload(config_entry.entry_id)

        assert _satellite(hass, config_entry) is None
