"""Tests for ensure_ascii=False in _convert_content_to_chat_message JSON serialization."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from custom_components.local_openai.entity import LocalAiEntity
from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm


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


@pytest.fixture
def base_entity(hass: HomeAssistant) -> LocalAiEntity:
    """Create a base LocalAiEntity for testing."""
    entry = MagicMock()
    entry.data = {}
    entry.runtime_data = MagicMock()

    subentry = MockSubentry({"model": "test-model"})

    entity = LocalAiEntity(entry, subentry)
    entity.hass = hass
    return entity


class TestToolResultNonAscii:
    """Tests for non-ASCII character handling in tool result serialization."""

    async def test_tool_result_preserves_cyrillic(self, base_entity: LocalAiEntity):
        """Test that Cyrillic characters are not escaped in tool results."""
        content = conversation.ToolResultContent(
            agent_id="test_agent",
            tool_call_id="call_1",
            tool_name="web_search",
            result=llm.ToolResult(
                data={"result": "Привет мир! Это русский текст с кириллицей."}
            ),
        )

        result = await base_entity._convert_content_to_chat_message(content)

        assert result is not None
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_1"
        assert "Привет мир" in result["content"]
        assert "\\u" not in result["content"]

    async def test_tool_result_preserves_cjk(self, base_entity: LocalAiEntity):
        """Test that CJK characters are not escaped in tool results."""
        content = conversation.ToolResultContent(
            agent_id="test_agent",
            tool_call_id="call_2",
            tool_name="weather",
            result=llm.ToolResult(
                data={"result": "东京の天気は晴れです。温度は25度です。"}
            ),
        )

        result = await base_entity._convert_content_to_chat_message(content)

        assert result is not None
        assert "东京" in result["content"]
        assert "\\u" not in result["content"]

    async def test_tool_result_preserves_arabic(self, base_entity: LocalAiEntity):
        """Test that Arabic characters are not escaped in tool results."""
        content = conversation.ToolResultContent(
            agent_id="test_agent",
            tool_call_id="call_3",
            tool_name="reminder",
            result=llm.ToolResult(data={"result": "مرحبا بالعالم! هذا نص عربي."}),
        )

        result = await base_entity._convert_content_to_chat_message(content)

        assert result is not None
        assert "مرحبا" in result["content"]
        assert "\\u" not in result["content"]

    async def test_tool_result_preserves_emoji(self, base_entity: LocalAiEntity):
        """Test that emoji characters are not escaped in tool results."""
        content = conversation.ToolResultContent(
            agent_id="test_agent",
            tool_call_id="call_4",
            tool_name="script",
            result=llm.ToolResult(
                data={
                    "result": "Success! The operation completed \u2705 with status 200."
                }
            ),
        )

        result = await base_entity._convert_content_to_chat_message(content)

        assert result is not None
        assert "\u2705" in result["content"]
        assert "\\u2705" not in result["content"]

    async def test_tool_result_mixed_ascii_and_non_ascii(
        self, base_entity: LocalAiEntity
    ):
        """Test that mixed ASCII and non-ASCII content is handled correctly."""
        content = conversation.ToolResultContent(
            agent_id="test_agent",
            tool_call_id="call_5",
            tool_name="web_search",
            result=llm.ToolResult(
                data={
                    "query": "weather",
                    "result": "The weather in Moscow (Москва) is sunny. Temperature: 20\u00b0C",
                }
            ),
        )

        result = await base_entity._convert_content_to_chat_message(content)

        assert result is not None
        assert "Москва" in result["content"]
        assert "weather" in result["content"]
        assert "\\u" not in result["content"]


class TestToolCallArgsNonAscii:
    """Tests for non-ASCII character handling in tool call argument serialization."""

    async def test_tool_call_args_preserves_cyrillic(self, base_entity: LocalAiEntity):
        """Test that Cyrillic characters are not escaped in tool call arguments."""
        content = conversation.AssistantContent(
            agent_id="test_agent",
            content=None,
            tool_calls=[
                llm.ToolInput(
                    id="call_1",
                    tool_name="web_search",
                    tool_args={"query": "погода в Москве"},
                    external=False,
                )
            ],
        )

        result = await base_entity._convert_content_to_chat_message(content)

        assert result is not None
        assert result["role"] == "assistant"
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1
        function_args = result["tool_calls"][0]["function"]["arguments"]
        assert "погода в Москве" in function_args
        assert "\\u" not in function_args

    async def test_tool_call_args_preserves_cjk(self, base_entity: LocalAiEntity):
        """Test that CJK characters are not escaped in tool call arguments."""
        content = conversation.AssistantContent(
            agent_id="test_agent",
            content=None,
            tool_calls=[
                llm.ToolInput(
                    id="call_2",
                    tool_name="script",
                    tool_args={"command": "echo こんにちは世界"},
                    external=False,
                )
            ],
        )

        result = await base_entity._convert_content_to_chat_message(content)

        assert result is not None
        assert "tool_calls" in result
        function_args = result["tool_calls"][0]["function"]["arguments"]
        assert "こんにちは世界" in function_args
        assert "\\u" not in function_args

    async def test_tool_call_args_mixed_content(self, base_entity: LocalAiEntity):
        """Test that mixed content in tool call args is handled correctly."""
        content = conversation.AssistantContent(
            agent_id="test_agent",
            content=None,
            tool_calls=[
                llm.ToolInput(
                    id="call_3",
                    tool_name="weather",
                    tool_args={
                        "location": "Москва, Россия",
                        "unit": "celsius",
                        "notes": "Примечание: дождь",
                    },
                    external=False,
                )
            ],
        )

        result = await base_entity._convert_content_to_chat_message(content)

        assert result is not None
        function_args = result["tool_calls"][0]["function"]["arguments"]
        assert "Москва" in function_args
        assert "дождь" in function_args
        assert "celsius" in function_args
        assert "\\u" not in function_args
