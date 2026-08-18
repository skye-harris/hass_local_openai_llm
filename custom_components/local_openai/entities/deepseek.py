"""Server-specific entities for DeepSeek Cloud."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

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
    from types import MappingProxyType

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
            ),
        ),
    }


def get_conversation_config_schema() -> dict:
    """Return conversation config schema fields for DeepSeek."""
    return _get_deepseek_schema()


def get_ai_task_config_schema() -> dict:
    """Return AI task config schema fields for DeepSeek."""
    return _get_deepseek_schema()


class DeepSeekMixin:
    """Mixin for DeepSeek entities with shared logic."""

    def _get_extra_body_args(self, options: MappingProxyType[str, Any]) -> dict:
        """Handle extra_body args for DeepSeek."""
        extra = super()._get_extra_body_args(options)
        opts = options.get(CONF_DEEPSEEK_CONFIG, {})
        reasoning_effort = opts.get(CONF_DEEPSEEK_REASONING_EFFORT)
        if reasoning_effort:
            extra["thinking"] = {"type": "enabled"}
            extra["reasoning_effort"] = reasoning_effort
        else:
            extra["thinking"] = {"type": "disabled"}
        return extra

    async def _convert_content_to_chat_message(
        self,
        content: conversation.Content,
    ) -> ChatCompletionMessageParam | None:
        """If thinking is enabled, pass prior thinking content back in the request."""
        param = await super()._convert_content_to_chat_message(content)
        opts = self.subentry.data.get(CONF_DEEPSEEK_CONFIG, {})

        if (
            opts.get(CONF_DEEPSEEK_REASONING_EFFORT)
            and isinstance(content, conversation.AssistantContent)
            and hasattr(content, "thinking_content")
            and content.thinking_content
        ):
            param["reasoning_content"] = content.thinking_content
        return param


class DeepSeekConversationEntity(DeepSeekMixin, LocalAiConversationEntity):
    """Conversation agent for DeepSeek Cloud servers."""


class DeepSeekAITaskEntity(DeepSeekMixin, LocalAITaskEntity):
    """AI Task entity for DeepSeek Cloud servers."""
