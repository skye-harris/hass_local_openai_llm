"""Server-specific entities for llama.cpp."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
)

from custom_components.local_openai.ai_task import LocalAITaskEntity
from custom_components.local_openai.const import (
    CONF_LLAMACPP_CONFIG,
    CONF_LLAMACPP_ENABLE_THINKING,
    CONF_LLAMACPP_ID_SLOT,
    CONF_LLAMACPP_INCLUDE_PRIOR_THINKING,
    CONF_LLAMACPP_MIN_P,
    CONF_LLAMACPP_PRESENCE_PENALTY,
    CONF_LLAMACPP_REPEAT_PENALTY,
    CONF_LLAMACPP_TOP_K,
    CONF_LLAMACPP_TOP_P,
)
from custom_components.local_openai.conversation import LocalAiConversationEntity

if TYPE_CHECKING:
    from types import MappingProxyType

    from openai.types.chat import ChatCompletionMessageParam

_LOGGER = logging.getLogger(__name__)


def get_model_alias(model: dict | object) -> str | None:
    """
    Return the alias llama.cpp exposes for a model, if one is set.

    llama.cpp exposes the value supplied via ``--alias`` as an extra ``alias`` field
    on the OpenAI-compatible model object. Returns ``None`` when no alias is set so
    the caller can fall back to (and clean up) the raw model ``id``.
    """
    return getattr(model, "alias", None)


def _get_llama_cpp_schema() -> dict:
    """llama.cpp server configuration schema."""
    return {
        vol.Required(CONF_LLAMACPP_ENABLE_THINKING, default=False): bool,
        vol.Required(CONF_LLAMACPP_INCLUDE_PRIOR_THINKING, default=True): bool,
        vol.Optional(CONF_LLAMACPP_ID_SLOT): NumberSelector(
            NumberSelectorConfig(min=0, step=1, mode=NumberSelectorMode.BOX),
        ),
        vol.Optional(
            CONF_LLAMACPP_TOP_P,
        ): NumberSelector(
            NumberSelectorConfig(
                min=0,
                max=1,
                step=0.05,
                mode=NumberSelectorMode.BOX,
            ),
        ),
        vol.Optional(
            CONF_LLAMACPP_TOP_K,
        ): NumberSelector(
            NumberSelectorConfig(
                min=1,
                max=1000,
                step=1,
                mode=NumberSelectorMode.BOX,
            ),
        ),
        vol.Optional(
            CONF_LLAMACPP_MIN_P,
        ): NumberSelector(
            NumberSelectorConfig(
                min=0,
                max=1,
                step=0.05,
                mode=NumberSelectorMode.BOX,
            ),
        ),
        vol.Optional(
            CONF_LLAMACPP_REPEAT_PENALTY,
        ): NumberSelector(
            NumberSelectorConfig(
                min=-2,
                max=2,
                step=0.05,
                mode=NumberSelectorMode.BOX,
            ),
        ),
        vol.Optional(
            CONF_LLAMACPP_PRESENCE_PENALTY,
        ): NumberSelector(
            NumberSelectorConfig(
                min=-2,
                max=2,
                step=0.05,
                mode=NumberSelectorMode.BOX,
            ),
        ),
    }


def get_conversation_config_schema() -> dict:
    """Return conversation config schema fields for llama.cpp."""
    return _get_llama_cpp_schema()


def get_ai_task_config_schema() -> dict:
    """Return AI task config schema fields for llama.cpp."""
    return _get_llama_cpp_schema()


class LlamaCppMixin:
    """Mixin for llama.cpp entities with shared logic."""

    def _get_extra_body_args(
        self,
        options: MappingProxyType[str, Any],
    ) -> dict:
        """Handle extra_body args for llama.cpp."""
        opts = options.get(CONF_LLAMACPP_CONFIG, {})
        extras: dict = {}

        id_slot = opts.get(CONF_LLAMACPP_ID_SLOT)
        if id_slot is not None:
            extras["id_slot"] = int(id_slot)

        extras["chat_template_kwargs"] = {
            "enable_thinking": bool(opts.get(CONF_LLAMACPP_ENABLE_THINKING, False)),
        }

        sampling_params = [
            (CONF_LLAMACPP_TOP_P, float, "top_p"),
            (CONF_LLAMACPP_TOP_K, int, "top_k"),
            (CONF_LLAMACPP_MIN_P, float, "min_p"),
            (CONF_LLAMACPP_REPEAT_PENALTY, float, "repeat_penalty"),
            (CONF_LLAMACPP_PRESENCE_PENALTY, float, "presence_penalty"),
        ]

        for conf_key, converter, arg_name in sampling_params:
            value = opts.get(conf_key)
            if value is not None:
                extras[arg_name] = converter(value)

        return extras

    async def _convert_content_to_chat_message(
        self,
        content: conversation.Content,
    ) -> ChatCompletionMessageParam | None:
        """If include_prior_reasoning is enabled, pass prior thinking content back in the request."""
        opts = self.subentry.data.get(CONF_LLAMACPP_CONFIG, {})
        param = await super()._convert_content_to_chat_message(content)

        if (
            opts.get(CONF_LLAMACPP_INCLUDE_PRIOR_THINKING, True)
            and isinstance(content, conversation.AssistantContent)
            and hasattr(content, "thinking_content")
            and content.thinking_content
        ):
            param["reasoning_content"] = content.thinking_content
        return param


class LlamaCppConversationEntity(LlamaCppMixin, LocalAiConversationEntity):
    """Conversation agent for llama.cpp servers."""


class LlamaCppAITaskEntity(LlamaCppMixin, LocalAITaskEntity):
    """AI Task entity for llama.cpp servers."""
