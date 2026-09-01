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
    CONF_LLAMACPP_USE_LOADED_MODEL,
)
from custom_components.local_openai.conversation import LocalAiConversationEntity

if TYPE_CHECKING:
    from types import MappingProxyType

    from homeassistant.config_entries import ConfigSubentry
    from openai.types.chat import ChatCompletionMessageParam

    from . import LocalAiConfigEntry

_LOGGER = logging.getLogger(__name__)

REQUEST_BODY_RESERVED_PARAMETERS = frozenset()
REQUEST_BODY_CONFIGURABLE_PARAMETERS = frozenset(
    {
        "id_slot",
        "top_p",
        "top_k",
        "min_p",
        "repeat_penalty",
        "presence_penalty",
    },
)


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
        vol.Required(CONF_LLAMACPP_USE_LOADED_MODEL, default=False): bool,
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

    def __init__(
        self,
        entry: LocalAiConfigEntry,
        subentry: ConfigSubentry,
    ) -> None:
        """Initialize the entity."""
        super().__init__(entry, subentry)

    async def _async_get_model(
        self,
        chat_log: conversation.ChatLog | None = None,
    ) -> str:
        """
        Return the model to use.

        When use_loaded_model is enabled, fetches fresh from the server's
        /v1/models endpoint every time. Prefers the configured model if it
        is loaded; falls back to the first loaded model; falls back to the
        configured model on any error.

        Modalities are detected from chat_log.content — if any content item
        has an image attachment, ["image"] is used to filter loaded models.
        Only models whose architecture.input_modalities contains all required
        modalities are considered.
        """
        opts = self.subentry.data.get(CONF_LLAMACPP_CONFIG, {})
        if not opts.get(CONF_LLAMACPP_USE_LOADED_MODEL, False):
            return self.model

        try:
            client = self.entry.runtime_data
            response = await client.models.list()

            loaded_models = [
                model
                for model in response.data
                if model.status.get("value") == "loaded"
            ]

            if not loaded_models:
                return self.model

            if chat_log is not None:
                modalities = [
                    "image"
                    for content in chat_log.content
                    if hasattr(content, "attachments")
                    and content.attachments
                    and any(
                        attachment.mime_type.startswith("image/")
                        for attachment in content.attachments
                    )
                ]

                if modalities:
                    loaded_models = [
                        model
                        for model in loaded_models
                        if set(modalities).issubset(
                            set(
                                getattr(model, "architecture", {}).get(
                                    "input_modalities", []
                                )
                            )
                        )
                    ]

                    if not loaded_models:
                        return self.model

            for model in loaded_models:
                if model.id == self.model:
                    return model.id

            return loaded_models[0].id
        except Exception:
            _LOGGER.exception("Failed to resolve loaded model, using configured model")
            return self.model

    def _get_extra_body_args(
        self,
        options: MappingProxyType[str, Any],
    ) -> dict:
        """Handle extra_body args for llama.cpp."""
        opts = options.get(CONF_LLAMACPP_CONFIG, {})
        extras = super()._get_extra_body_args(options)

        id_slot = opts.get(CONF_LLAMACPP_ID_SLOT)
        if id_slot is not None:
            extras["id_slot"] = int(id_slot)

        chat_template_kwargs = extras.get("chat_template_kwargs", {})
        chat_template_kwargs["enable_thinking"] = bool(
            opts.get(CONF_LLAMACPP_ENABLE_THINKING, False)
        )
        extras["chat_template_kwargs"] = chat_template_kwargs

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
        conversation_id: str | None = None,
    ) -> ChatCompletionMessageParam | None:
        """If include_prior_reasoning is enabled, pass prior thinking content back in the request."""
        opts = self.subentry.data.get(CONF_LLAMACPP_CONFIG, {})
        param = await super()._convert_content_to_chat_message(content, conversation_id)

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
