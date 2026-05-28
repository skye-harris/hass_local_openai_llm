"""Server-specific entities for llama.cpp."""

from __future__ import annotations

import logging

import voluptuous as vol
from homeassistant.helpers import template
from homeassistant.helpers.selector import ObjectSelector

from custom_components.local_openai.ai_task import LocalAITaskEntity
from custom_components.local_openai.const import (
    CONF_LLAMACPP_CONFIG,
    CONF_LLAMACPP_SERVER_KWARGS,
)
from custom_components.local_openai.conversation import LocalAiConversationEntity

_LOGGER = logging.getLogger(__name__)


def _get_llama_cpp_schema() -> dict:
    return {
        vol.Optional(CONF_LLAMACPP_SERVER_KWARGS, default=[]): ObjectSelector(
            config={
                "multiple": True,
                "fields": {
                    "Key": {
                        "selector": {"text": None},
                        "required": True,
                    },
                    "Value": {
                        "selector": {"template": None},
                        "required": True,
                    },
                },
            }
        ),
    }


def get_conversation_config_schema() -> dict:
    """Return conversation config schema fields for llama.cpp."""
    return _get_llama_cpp_schema()


def get_ai_task_config_schema() -> dict:
    """Return AI task config schema fields for llama.cpp."""
    return _get_llama_cpp_schema()


def _llama_cpp_extra_body_args(hass, options: dict) -> dict:
    opts = options.get(CONF_LLAMACPP_CONFIG, {})
    server_kwargs = opts.get(CONF_LLAMACPP_SERVER_KWARGS, [])

    rendered: dict = {}
    for keypair in server_kwargs:
        key = (keypair.get("Key") or "").strip()
        if not key:
            continue
        rendered[key] = template.Template(
            keypair.get("Value", ""),
            hass,
        ).async_render()
    return rendered


class LlamaCppConversationEntity(LocalAiConversationEntity):
    """Conversation agent for llama.cpp servers."""

    def _get_extra_body_args(self, options: dict, server_options: dict) -> dict:
        return _llama_cpp_extra_body_args(self.hass, options)


class LlamaCppAITaskEntity(LocalAITaskEntity):
    """AI Task entity for llama.cpp servers."""

    def _get_extra_body_args(self, options: dict, server_options: dict) -> dict:
        return _llama_cpp_extra_body_args(self.hass, options)
