"""Unit tests for Google Gemini content conversion with thought signatures."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from custom_components.local_openai.entities.google_gemini import (
    GoogleGeminiConversationEntity,
)
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
def google_entity(hass: HomeAssistant) -> GoogleGeminiConversationEntity:
    """Create a Google Gemini conversation entity for testing."""
    entry = MagicMock()
    entry.data = {}
    entry.runtime_data = MagicMock()

    subentry = MockSubentry({"model": "test-model"})

    entity = GoogleGeminiConversationEntity(entry, subentry)
    entity.hass = hass
    return entity


class TestConvertContentWithSignatures:
    """Tests for _convert_content_to_chat_message with thought signatures."""

    async def test_signature_stored_and_retrieved(self, google_entity):
        """Test storing and retrieving a thought signature."""
        conversation_id = "conv_123"
        tool_call_id = "call_abc"
        signature = "sig_xyz"

        google_entity._store_thought_signature(conversation_id, tool_call_id, signature)
        assert (
            google_entity._get_thought_signature(conversation_id, tool_call_id)
            == signature
        )

    async def test_signature_not_found_returns_none(self, google_entity):
        """Test that missing signature returns None."""
        assert google_entity._get_thought_signature("conv_1", "call_1") is None

    async def test_different_conversations_no_leak(self, google_entity):
        """Test that signatures don't leak between conversation IDs."""
        google_entity._store_thought_signature("conv_1", "call_1", "sig_1")
        google_entity._store_thought_signature("conv_2", "call_1", "sig_2")

        assert google_entity._get_thought_signature("conv_1", "call_1") == "sig_1"
        assert google_entity._get_thought_signature("conv_2", "call_1") == "sig_2"

    async def test_convert_with_all_signatures_reattaches_extra_content(
        self, google_entity
    ):
        """Test that AssistantContent with all signed tool_calls gets extra_content."""
        conversation_id = "conv_1"
        tool_call = llm.ToolInput(
            id="call_1",
            tool_name="test_fn",
            tool_args={"arg": "val"},
        )

        google_entity._store_thought_signature(conversation_id, "call_1", "sig_xyz")

        content = conversation.AssistantContent(
            agent_id="test_agent",
            content="thinking...",
            thinking_content="reasoning here",
            tool_calls=[tool_call],
        )

        result = await google_entity._convert_content_to_chat_message(
            content, conversation_id
        )

        assert result is not None
        assert result["role"] == "assistant"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["extra_content"] == {
            "google": {"thought_signature": "sig_xyz"}
        }
        assert result["thinking_content"] == "reasoning here"

    async def test_convert_with_missing_signature_drops_thinking(self, google_entity):
        """Test that missing signature causes thinking_content to be dropped."""
        conversation_id = "conv_1"
        tool_call = llm.ToolInput(
            id="call_1",
            tool_name="test_fn",
            tool_args={"arg": "val"},
        )

        # No signature stored for call_1

        content = conversation.AssistantContent(
            agent_id="test_agent",
            content="thinking...",
            thinking_content="reasoning here",
            tool_calls=[tool_call],
        )

        result = await google_entity._convert_content_to_chat_message(
            content, conversation_id
        )

        assert result is not None
        assert "thinking_content" not in result
        assert "extra_content" not in result["tool_calls"][0]

    async def test_convert_with_any_missing_signature_drops_thinking(
        self, google_entity
    ):
        """Test that if ANY tool_call lacks signature, thinking is dropped."""
        conversation_id = "conv_1"
        tool_call_1 = llm.ToolInput(
            id="call_1",
            tool_name="fn1",
            tool_args={},
        )
        tool_call_2 = llm.ToolInput(
            id="call_2",
            tool_name="fn2",
            tool_args={},
        )

        # Only sign call_1, not call_2

        google_entity._store_thought_signature(conversation_id, "call_1", "sig_1")

        content = conversation.AssistantContent(
            agent_id="test_agent",
            content="thinking...",
            thinking_content="reasoning here",
            tool_calls=[tool_call_1, tool_call_2],
        )

        result = await google_entity._convert_content_to_chat_message(
            content, conversation_id
        )

        assert result is not None
        assert "thinking_content" not in result

    async def test_convert_none_conversation_id_drops_thinking(self, google_entity):
        """Test that None conversation_id causes thinking to be dropped."""
        tool_call = llm.ToolInput(
            id="call_1",
            tool_name="test_fn",
            tool_args={"arg": "val"},
        )

        google_entity._store_thought_signature("conv_1", "call_1", "sig_xyz")

        content = conversation.AssistantContent(
            agent_id="test_agent",
            content="thinking...",
            thinking_content="reasoning here",
            tool_calls=[tool_call],
        )

        result = await google_entity._convert_content_to_chat_message(content, None)

        assert result is not None
        assert "thinking_content" not in result

    async def test_convert_assistant_without_tool_calls_unchanged(self, google_entity):
        """Test that assistant messages without tool_calls pass through."""
        content = conversation.AssistantContent(
            agent_id="test_agent",
            content="hello there",
        )

        result = await google_entity._convert_content_to_chat_message(content, "conv_1")

        assert result is not None
        assert result["role"] == "assistant"
        assert result["content"] == "hello there"
        assert "tool_calls" not in result

    async def test_convert_tool_result_passes_through(self, google_entity):
        """Test that tool result content passes through unchanged."""
        content = conversation.ToolResultContent(
            agent_id="test_agent",
            tool_call_id="call_1",
            tool_name="test_fn",
            tool_result={"result": "ok"},
        )

        result = await google_entity._convert_content_to_chat_message(content, "conv_1")

        assert result is not None
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_1"

    async def test_multiple_tool_calls_all_signed(self, google_entity):
        """Test multiple tool calls all get extra_content when all signed."""
        conversation_id = "conv_1"
        tool_call_1 = llm.ToolInput(
            id="call_1",
            tool_name="fn1",
            tool_args={"a": 1},
        )
        tool_call_2 = llm.ToolInput(
            id="call_2",
            tool_name="fn2",
            tool_args={"b": 2},
        )

        google_entity._store_thought_signature(conversation_id, "call_1", "sig_1")
        google_entity._store_thought_signature(conversation_id, "call_2", "sig_2")

        content = conversation.AssistantContent(
            agent_id="test_agent",
            content="",
            thinking_content="reasoning",
            tool_calls=[tool_call_1, tool_call_2],
        )

        result = await google_entity._convert_content_to_chat_message(
            content, conversation_id
        )

        assert result is not None
        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0]["extra_content"] == {
            "google": {"thought_signature": "sig_1"}
        }
        assert result["tool_calls"][1]["extra_content"] == {
            "google": {"thought_signature": "sig_2"}
        }
        assert result["thinking_content"] == "reasoning"
