"""Server-specific entities for llama.cpp."""

from __future__ import annotations

import logging

import voluptuous as vol
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
)
from custom_components.local_openai.conversation import LocalAiConversationEntity

_LOGGER = logging.getLogger(__name__)


def get_model_alias(model) -> str | None:
    """Return the alias llama.cpp exposes for a model, if one is set.

    llama.cpp exposes the value supplied via ``--alias`` as an extra ``alias`` field
    on the OpenAI-compatible model object. Returns ``None`` when no alias is set so
    the caller can fall back to (and clean up) the raw model ``id``.
    """
    return getattr(model, "alias", None)


def _get_llama_cpp_schema() -> dict:
    return {
        vol.Required(CONF_LLAMACPP_ENABLE_THINKING, default=False): bool,
        vol.Optional(CONF_LLAMACPP_ID_SLOT): NumberSelector(
            NumberSelectorConfig(min=0, step=1, mode=NumberSelectorMode.BOX)
        ),
    }


def get_conversation_config_schema() -> dict:
    """Return conversation config schema fields for llama.cpp."""
    return _get_llama_cpp_schema()


def get_ai_task_config_schema() -> dict:
    """Return AI task config schema fields for llama.cpp."""
    return _get_llama_cpp_schema()


def _llama_cpp_extra_body_args(options: dict) -> dict:
    opts = options.get(CONF_LLAMACPP_CONFIG, {})
    extras: dict = {}

    id_slot = opts.get(CONF_LLAMACPP_ID_SLOT)
    if id_slot is not None:
        extras["id_slot"] = int(id_slot)

    extras["chat_template_kwargs"] = {
        "enable_thinking": bool(opts.get(CONF_LLAMACPP_ENABLE_THINKING, False)),
    }

    return extras


class LlamaCppConversationEntity(LocalAiConversationEntity):
    """Conversation agent for llama.cpp servers."""

    def _get_extra_body_args(self, options: dict, server_options: dict) -> dict:
        return _llama_cpp_extra_body_args(options)


class LlamaCppAITaskEntity(LocalAITaskEntity):
    """AI Task entity for llama.cpp servers."""

    def _get_extra_body_args(self, options: dict, server_options: dict) -> dict:
        return _llama_cpp_extra_body_args(options)
