"""Tests for always_continue_conversation in LocalAiConversationEntity."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components import conversation
from homeassistant.core import HomeAssistant

from custom_components.local_openai.const import CONF_ALWAYS_CONTINUE_CONVERSATION
from custom_components.local_openai.conversation import LocalAiConversationEntity


@pytest.mark.parametrize(
    ("always_continue_conversation", "expected"),
    [
        pytest.param(True, True, id="enabled"),
        pytest.param(False, False, id="disabled"),
        pytest.param(None, False, id="missing_defaults_to_false"),
    ],
)
async def test_always_continue_conversation_sets_continue_conversation(
    hass: HomeAssistant,
    always_continue_conversation: bool | None,
    expected: bool,
) -> None:
    """Test always_continue_conversation controls continue_conversation on the result."""
    entry = MagicMock()
    entry.data = {}
    entry.runtime_data = MagicMock()

    data: dict[str, object] = {"model": "test-model"}
    if always_continue_conversation is not None:
        data[CONF_ALWAYS_CONTINUE_CONVERSATION] = always_continue_conversation
    subentry = MagicMock()
    subentry.data = data

    entity = LocalAiConversationEntity(entry, subentry)
    entity.hass = hass

    chat_log = MagicMock(spec=conversation.ChatLog)
    chat_log.content = [conversation.AssistantContent(agent_id="test", content="test")]
    chat_log.conversation_id = "test_conversation_id"
    chat_log.continue_conversation = False
    chat_log.llm_input_provided_index = 0
    chat_log.async_provide_llm_data = AsyncMock()
    entity._async_handle_chat_log = AsyncMock()

    with patch(
        "custom_components.local_openai.conversation.llm.async_get_apis",
        return_value=[],
    ):
        result = await entity._async_handle_message(
            MagicMock(spec=conversation.ConversationInput),
            chat_log,
        )

    assert result.continue_conversation is expected
