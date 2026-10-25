"""Tests for StopTalkingTool and its integration with LocalAiConversationEntity."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from custom_components.local_openai.conversation import LocalAiConversationEntity
from custom_components.local_openai.tools.stop_talking import StopTalkingTool
from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm


class TestStopTalkingTool:
    """Test StopTalkingTool behavior."""

    async def test_returns_empty_tool_result(self) -> None:
        """Test that StopTalkingTool returns an empty ToolResult when called."""
        tool = StopTalkingTool()
        hass = MagicMock()
        tool_input = MagicMock(spec=llm.ToolInput)
        llm_context = MagicMock(spec=llm.LLMContext)

        result = await tool.async_call(hass, tool_input, llm_context)

        assert isinstance(result, llm.ToolResult)
        assert result.data == {}


class TestStopTalkingSetsContinueConversation:
    """Test that stop_talking detection sets continue_conversation to False."""

    async def test_stop_talking_sets_continue_conversation_false(
        self,
        hass: HomeAssistant,
    ) -> None:
        """Test that stop_talking detection sets continue_conversation to False."""
        entry = MagicMock()
        entry.data = {}
        entry.runtime_data = MagicMock()

        subentry = MagicMock()
        subentry.data = {"model": "test-model"}

        entity = LocalAiConversationEntity(entry, subentry)
        entity.hass = hass

        chat_log = MagicMock(spec=conversation.ChatLog)
        chat_log.content = [
            conversation.AssistantContent(agent_id="test", content="test")
        ]
        chat_log.conversation_id = "test_conversation_id"
        chat_log.continue_conversation = False
        chat_log.llm_input_provided_index = 0
        chat_log.async_provide_llm_data = AsyncMock()

        entity._async_handle_chat_log = AsyncMock(return_value=True)

        with patch(
            "custom_components.local_openai.conversation.llm.async_get_apis",
            return_value=[],
        ):
            result = await entity._async_handle_message(
                MagicMock(spec=conversation.ConversationInput),
                chat_log,
            )

        assert result.continue_conversation is False

    async def test_stop_talking_overrides_always_continue_conversation(
        self,
        hass: HomeAssistant,
    ) -> None:
        """Test that stop_talking sets continue_conversation to False even when always_continue is True."""
        entry = MagicMock()
        entry.data = {}
        entry.runtime_data = MagicMock()

        subentry = MagicMock()
        subentry.data = {
            "model": "test-model",
            "always_continue_conversation": True,
        }

        entity = LocalAiConversationEntity(entry, subentry)
        entity.hass = hass

        chat_log = MagicMock(spec=conversation.ChatLog)
        chat_log.content = [
            conversation.AssistantContent(agent_id="test", content="test")
        ]
        chat_log.conversation_id = "test_conversation_id"
        chat_log.continue_conversation = False
        chat_log.llm_input_provided_index = 0
        chat_log.async_provide_llm_data = AsyncMock()

        entity._async_handle_chat_log = AsyncMock(return_value=True)

        with patch(
            "custom_components.local_openai.conversation.llm.async_get_apis",
            return_value=[],
        ):
            result = await entity._async_handle_message(
                MagicMock(spec=conversation.ConversationInput),
                chat_log,
            )

        assert result.continue_conversation is False

    async def test_no_stop_talking_uses_always_continue_conversation(
        self,
        hass: HomeAssistant,
    ) -> None:
        """Test that without stop_talking, always_continue_conversation is respected."""
        entry = MagicMock()
        entry.data = {}
        entry.runtime_data = MagicMock()

        subentry = MagicMock()
        subentry.data = {
            "model": "test-model",
            "always_continue_conversation": True,
        }

        entity = LocalAiConversationEntity(entry, subentry)
        entity.hass = hass

        chat_log = MagicMock(spec=conversation.ChatLog)
        chat_log.content = [
            conversation.AssistantContent(agent_id="test", content="test")
        ]
        chat_log.conversation_id = "test_conversation_id"
        chat_log.continue_conversation = False
        chat_log.llm_input_provided_index = 0
        chat_log.async_provide_llm_data = AsyncMock()

        entity._async_handle_chat_log = AsyncMock(return_value=False)

        with patch(
            "custom_components.local_openai.conversation.llm.async_get_apis",
            return_value=[],
        ):
            result = await entity._async_handle_message(
                MagicMock(spec=conversation.ConversationInput),
                chat_log,
            )

        assert result.continue_conversation is True
