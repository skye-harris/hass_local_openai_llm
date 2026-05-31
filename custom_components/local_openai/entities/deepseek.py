"""Server-specific entities for DeepSeek Cloud."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.helpers.selector import (
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
)

from custom_components.local_openai.ai_task import LocalAITaskEntity
from custom_components.local_openai.const import (
    CONF_DEEPSEEK_CONFIG,
    CONF_DEEPSEEK_REASONING_EFFORT,
)
from custom_components.local_openai.conversation import LocalAiConversationEntity

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigSubentry
    from openai.types.chat import ChatCompletionMessageParam

_LOGGER = logging.getLogger(__name__)


def _get_deepseek_schema() -> dict:
    """DeepSeek server configuration schema."""
    return {
        vol.Optional(
            CONF_DEEPSEEK_REASONING_EFFORT,
        ): SelectSelector(
            SelectSelectorConfig(
                options=[
                    SelectOptionDict(value="high", label="High"),
                    SelectOptionDict(value="max", label="Max"),
                ],
                mode=SelectSelectorMode.DROPDOWN,
            )
        ),
    }


def get_conversation_config_schema() -> dict:
    """Return conversation config schema fields for DeepSeek."""
    return _get_deepseek_schema()


def _deepseek_extra_body_args(options: dict) -> dict:
    """Handle extra_body args for DeepSeek."""
    opts = options.get(CONF_DEEPSEEK_CONFIG, {})
    reasoning_effort = opts.get(CONF_DEEPSEEK_REASONING_EFFORT)
    if reasoning_effort:
        return {
            "thinking": {"type": "enabled"},
            "reasoning_effort": reasoning_effort,
        }
    return {"thinking": {"type": "disabled"}}


async def _deepseek_augment_content_message(
    subentry: ConfigSubentry,
    param: ChatCompletionMessageParam | None,
    content: conversation.Content,
) -> ChatCompletionMessageParam | None:
    """If thinking is enabled, and the message has thinking content, pass this back in the request."""
    opts = subentry.data.get(CONF_DEEPSEEK_CONFIG, {})

    if (
        opts.get(CONF_DEEPSEEK_REASONING_EFFORT)
        and isinstance(content, conversation.AssistantContent)
        and hasattr(content, "thinking_content")
        and content.thinking_content
    ):
        param["reasoning_content"] = content.thinking_content
    return param


class DeepSeekConversationEntity(LocalAiConversationEntity):
    """Conversation agent for DeepSeek Cloud servers."""

    def _get_extra_body_args(self, options: dict, server_options: dict) -> dict:
        """Handle extra arguments for DeepSeek."""
        return _deepseek_extra_body_args(options)

    async def _convert_content_to_chat_message(
        self, content: conversation.Content
    ) -> ChatCompletionMessageParam | None:
        """Handle chat message conversion for DeepSeek."""
        param = await super()._convert_content_to_chat_message(content)
        return await _deepseek_augment_content_message(self.subentry, param, content)


class DeepSeekAITaskEntity(LocalAITaskEntity):
    """AI Task entity for DeepSeek Cloud servers."""

    def _get_extra_body_args(self, options: dict, server_options: dict) -> dict:
        """Handle extra arguments for DeepSeek."""
        return _deepseek_extra_body_args(options)

    async def _convert_content_to_chat_message(
        self, content: conversation.Content
    ) -> ChatCompletionMessageParam | None:
        """Handle chat message conversion for DeepSeek."""
        param = await super()._convert_content_to_chat_message(content)
        return await _deepseek_augment_content_message(self.subentry, param, content)
