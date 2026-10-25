"""Server-specific entities for vLLM."""

from __future__ import annotations

from typing import TYPE_CHECKING

import voluptuous as vol
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
)

from custom_components.local_openai.ai_task import LocalAITaskEntity
from custom_components.local_openai.const import (
    CONF_VLLM_CONFIG,
    CONF_VLLM_THINKING_MODE,
    CONF_VLLM_THINKING_MODE_ALWAYS,
    CONF_VLLM_THINKING_MODE_FIRST_LOOP,
    CONF_VLLM_THINKING_TOKEN_BUDGET,
    CONF_VLLM_THINKING_TOKEN_BUDGET_DEFAULT,
)
from custom_components.local_openai.conversation import LocalAiConversationEntity

if TYPE_CHECKING:
    from custom_components.local_openai.entity import AgentContext

REQUEST_BODY_RESERVED_PARAMETERS = frozenset()
REQUEST_BODY_CONFIGURABLE_PARAMETERS = frozenset({"thinking_token_budget"})


def _get_vllm_schema() -> dict:
    """Return the vLLM server configuration schema."""
    return {
        vol.Optional(CONF_VLLM_THINKING_MODE): SelectSelector(
            SelectSelectorConfig(
                options=[
                    SelectOptionDict(
                        value=CONF_VLLM_THINKING_MODE_ALWAYS,
                        label="Always on",
                    ),
                    SelectOptionDict(
                        value=CONF_VLLM_THINKING_MODE_FIRST_LOOP,
                        label="First loop iteration only",
                    ),
                ],
                mode=SelectSelectorMode.DROPDOWN,
            ),
        ),
        vol.Required(
            CONF_VLLM_THINKING_TOKEN_BUDGET,
            default=CONF_VLLM_THINKING_TOKEN_BUDGET_DEFAULT,
        ): NumberSelector(
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

    def _get_extra_body_args(self, context: AgentContext) -> None:
        """Add the vLLM thinking token budget and enable_thinking to the base extra_body args."""
        options = self.options
        super()._get_extra_body_args(context)
        extra_body_args = context.extra_body

        opts = options.get(CONF_VLLM_CONFIG, {})
        thinking_mode = opts.get(CONF_VLLM_THINKING_MODE)
        thinking_enabled = bool(thinking_mode) and not (
            thinking_mode == CONF_VLLM_THINKING_MODE_FIRST_LOOP
            and context.iteration > 0
        )

        thinking_token_budget = opts.get(
            CONF_VLLM_THINKING_TOKEN_BUDGET,
            CONF_VLLM_THINKING_TOKEN_BUDGET_DEFAULT,
        )
        if thinking_enabled:
            extra_body_args["thinking_token_budget"] = int(thinking_token_budget)
        else:
            extra_body_args.pop("thinking_token_budget", None)

        chat_template_kwargs = extra_body_args.setdefault("chat_template_kwargs", {})
        chat_template_kwargs["enable_thinking"] = thinking_enabled


class VllmConversationEntity(VllmMixin, LocalAiConversationEntity):
    """Conversation agent for vLLM servers."""


class VllmAITaskEntity(VllmMixin, LocalAITaskEntity):
    """AI Task entity for vLLM servers."""
