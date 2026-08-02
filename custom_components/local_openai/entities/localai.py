"""Server-specific entities for LocalAI."""

from __future__ import annotations

from typing import Any

from custom_components.local_openai.ai_task import LocalAITaskEntity
from custom_components.local_openai.conversation import LocalAiConversationEntity


def _to_metadata_value(value: Any) -> str:
    """
    LocalAI's per-request metadata field is string-only.

    See https://localai.io/advanced/model-configuration/index.html#custom-chat_template_kwargs
    """
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


class LocalAIServerMixin:
    """
    Mixin for LocalAI entities.

    LocalAI does not read a top-level ``chat_template_kwargs`` field the way
    llama.cpp's server does. Chat template variables are supplied via the
    OpenAI ``metadata`` field, with string values.

    See https://localai.io/advanced/model-configuration/index.html#custom-chat_template_kwargs
    """

    def _get_extra_body_args(self, options: dict) -> dict:
        """Route chat template kwargs into `metadata` as strings for LocalAI."""
        extra_body_args = super()._get_extra_body_args(options)

        kwargs = extra_body_args.pop("chat_template_kwargs", None)
        if kwargs:
            metadata = extra_body_args.setdefault("metadata", {})
            for key, value in kwargs.items():
                metadata[key] = _to_metadata_value(value)

        return extra_body_args


class LocalAIServerConversationEntity(
    LocalAIServerMixin,
    LocalAiConversationEntity,
):
    """Conversation agent for LocalAI servers."""


class LocalAIServerAITaskEntity(LocalAIServerMixin, LocalAITaskEntity):
    """AI Task entity for LocalAI servers."""
