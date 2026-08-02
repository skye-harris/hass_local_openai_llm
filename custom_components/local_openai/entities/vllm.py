"""Server-specific entities for vLLM."""

from __future__ import annotations

import voluptuous as vol
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
)

from custom_components.local_openai.ai_task import LocalAITaskEntity
from custom_components.local_openai.const import (
    CONF_VLLM_CONFIG,
    CONF_VLLM_THINKING_TOKEN_BUDGET,
)
from custom_components.local_openai.conversation import LocalAiConversationEntity


def _get_vllm_schema() -> dict:
    """Return the vLLM server configuration schema."""
    return {
        vol.Optional(CONF_VLLM_THINKING_TOKEN_BUDGET): NumberSelector(
            NumberSelectorConfig(
                min=0,
                max=32768,
                step=1,
                mode=NumberSelectorMode.BOX,
            ),
        ),
    }


def get_conversation_config_schema() -> dict:
    """Return conversation config schema fields for vLLM."""
    return _get_vllm_schema()


def get_ai_task_config_schema() -> dict:
    """Return AI task config schema fields for vLLM."""
    return _get_vllm_schema()


class VllmMixin:
    """Mixin for vLLM entities with shared logic."""

    def _get_extra_body_args(self, options: dict) -> dict:
        """Add the vLLM thinking token budget to the base extra_body args."""
        extra_body_args = super()._get_extra_body_args(options)

        thinking_token_budget = options.get(CONF_VLLM_CONFIG, {}).get(
            CONF_VLLM_THINKING_TOKEN_BUDGET,
        )
        if thinking_token_budget is not None:
            extra_body_args["thinking_token_budget"] = int(thinking_token_budget)

        return extra_body_args


class VllmConversationEntity(VllmMixin, LocalAiConversationEntity):
    """Conversation agent for vLLM servers."""


class VllmAITaskEntity(VllmMixin, LocalAITaskEntity):
    """AI Task entity for vLLM servers."""
