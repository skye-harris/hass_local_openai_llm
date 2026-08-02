"""Unit tests for _convert_content_to_chat_message override with thinking content."""

from __future__ import annotations

import pytest
from types import MappingProxyType

from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_MODEL

from tests.conftest import MockAssistantContent

from custom_components.local_openai.const import (
    CONF_LLAMACPP_CONFIG,
    CONF_LLAMACPP_ENABLE_THINKING,
    CONF_LLAMACPP_INCLUDE_PRIOR_THINKING,
)
from custom_components.local_openai.entities.llama_cpp import LlamaCppConversationEntity


@pytest.mark.parametrize(
    "options,content,expect_reasoning",
    [
        (
            {
                CONF_MODEL: "test-model",
                CONF_LLAMACPP_CONFIG: {
                    CONF_LLAMACPP_ENABLE_THINKING: True,
                    CONF_LLAMACPP_INCLUDE_PRIOR_THINKING: True,
                },
            },
            MockAssistantContent(thinking_content="thinking steps here"),
            "thinking steps here",
        ),
        (
            {
                CONF_MODEL: "test-model",
                CONF_LLAMACPP_CONFIG: {
                    CONF_LLAMACPP_ENABLE_THINKING: True,
                    CONF_LLAMACPP_INCLUDE_PRIOR_THINKING: True,
                },
            },
            MockAssistantContent(thinking_content=None),
            None,
        ),
        (
            {
                CONF_MODEL: "test-model",
                CONF_LLAMACPP_CONFIG: {
                    CONF_LLAMACPP_ENABLE_THINKING: True,
                    CONF_LLAMACPP_INCLUDE_PRIOR_THINKING: False,
                },
            },
            MockAssistantContent(thinking_content="thinking steps here"),
            None,
        ),
        (
            {
                CONF_MODEL: "test-model",
                CONF_LLAMACPP_CONFIG: {
                    CONF_LLAMACPP_ENABLE_THINKING: False,
                    CONF_LLAMACPP_INCLUDE_PRIOR_THINKING: True,
                },
            },
            MockAssistantContent(thinking_content="thinking steps here"),
            "thinking steps here",
        ),
    ],
)
async def test_llama_cpp_convert_content_to_chat_message(
    mock_config_entry_llamacpp,
    options,
    content,
    expect_reasoning,
):
    entity = LlamaCppConversationEntity(
        mock_config_entry_llamacpp,
        ConfigSubentry(
            subentry_id="test_conversation_subentry_id",
            subentry_type="conversation",
            title="Conversation Agent",
            data=MappingProxyType(options),
            unique_id=None,
        ),
    )
    result = await entity._convert_content_to_chat_message(content)
    assert result.get("reasoning_content") == expect_reasoning
