"""Tests that tool schemas are sent in a stable order.

Home Assistant builds the tool list from intent.async_get(), whose order is the order
integrations registered their intents during startup -- a race that re-rolls on every
restart. Chat templates render tool schemas at the start of the prompt, so an unstable
order invalidates the whole prefix of a server-side prompt cache.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import voluptuous as vol
from homeassistant.const import CONF_MODEL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from custom_components.local_openai.conversation import LocalAiConversationEntity


class MockSubentry:
    """Mock ConfigSubentry that allows updating data."""

    def __setattr__(self, name: str, value: object) -> None:
        object.__setattr__(self, name, value)

    def __init__(self, data: dict) -> None:
        self.subentry_id = "test_conversation_subentry_id"
        self.subentry_type = "conversation"
        self.title = "Conversation Agent"
        self.data = data
        self.unique_id = None


class StubTool(llm.Tool):
    """Minimal llm.Tool with just enough for _format_tool."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"{name} description"
        self.parameters = vol.Schema({})

    async def async_call(self, hass, tool_input, llm_context):  # noqa: ANN001, ANN201
        """Not exercised by these tests."""
        return {}


@pytest.fixture
def mock_entity(hass: HomeAssistant) -> LocalAiConversationEntity:
    """Create a conversation entity with no content injection configured."""
    entry = MagicMock()
    entry.data = {}
    entry.runtime_data = MagicMock()

    entity = LocalAiConversationEntity(entry, MockSubentry({CONF_MODEL: "test-model"}))
    entity.hass = hass
    return entity


def _chat_log(tool_names: list[str]) -> MagicMock:
    chat_log = MagicMock()
    chat_log.llm_api.tools = [StubTool(name) for name in tool_names]
    chat_log.llm_api.custom_serializer = None
    chat_log.content = []
    return chat_log


async def _captured_tool_names(entity, chat_log) -> list[str]:
    entity._run_agent_loop = AsyncMock()  # noqa: SLF001

    await entity._async_handle_chat_log(chat_log)  # noqa: SLF001

    model_args = entity._run_agent_loop.call_args.args[1]  # noqa: SLF001
    return [tool["function"]["name"] for tool in model_args["tools"]]


async def test_tools_are_sorted_by_name(mock_entity) -> None:
    """Tools are sent alphabetically, not in the order the API happened to yield."""
    chat_log = _chat_log(
        ["HassTurnOn", "GetLiveContext", "HassLightSet", "HassBroadcast"]
    )

    assert await _captured_tool_names(mock_entity, chat_log) == [
        "GetLiveContext",
        "HassBroadcast",
        "HassLightSet",
        "HassTurnOn",
    ]


async def test_tool_order_is_independent_of_input_order(mock_entity) -> None:
    """The same set of tools produces the same wire order however it arrives.

    This is the property that matters: it is what lets a server-side prompt cache
    survive a Home Assistant restart.
    """
    names = ["HassTurnOn", "GetLiveContext", "HassLightSet", "HassBroadcast"]
    first = await _captured_tool_names(mock_entity, _chat_log(names))
    second = await _captured_tool_names(mock_entity, _chat_log(list(reversed(names))))

    assert first == second
