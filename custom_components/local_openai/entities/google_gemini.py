"""Server-specific entities for Google Gemini."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from homeassistant.components import conversation
from homeassistant.helpers.entity import Entity

from custom_components.local_openai.ai_task import LocalAITaskEntity
from custom_components.local_openai.conversation import LocalAiConversationEntity

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam


_LOGGER = logging.getLogger(__name__)


def get_conversation_config_schema() -> dict:
    """Return conversation config schema fields for Google Gemini."""
    return {}


def get_ai_task_config_schema() -> dict:
    """Return AI task config schema fields for Google Gemini."""
    return {}


class GoogleGeminiMixin(Entity):
    """Mixin for Google Gemini entities — manages thought signature cache."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the mixin."""
        self._thought_signatures: dict[str, dict[str, str]] = {}
        super().__init__(*args, **kwargs)

    def _store_thought_signature(
        self,
        conversation_id: str,
        tool_call_id: str,
        signature: str,
    ) -> None:
        """Store a thought signature keyed by conversation + tool call."""
        self._thought_signatures.setdefault(conversation_id, {})[tool_call_id] = (
            signature
        )

    def _get_thought_signature(
        self,
        conversation_id: str,
        tool_call_id: str,
    ) -> str | None:
        """Retrieve a thought signature, or None if not cached."""
        return self._thought_signatures.get(conversation_id, {}).get(tool_call_id)

    def _on_tool_call_delta(
        self,
        tool_call_id: str | None,
        _tool_call_name: str | None,
        tool_call: Any,
        conversation_id: str | None = None,
    ) -> None:
        """Capture thought_signature from streaming tool call deltas."""
        if conversation_id is None:
            return

        try:
            google_data = tool_call.extra_content.get("google", {})
            sig = google_data.get("thought_signature")
            if sig:
                self._store_thought_signature(conversation_id, tool_call_id, sig)
        except AttributeError:
            pass

    async def _convert_content_to_chat_message(
        self,
        content: conversation.Content,
        conversation_id: str | None = None,
    ) -> ChatCompletionMessageParam | None:
        """Re-attach extra_content from cache; drop thinking if signature missing."""
        param = await super()._convert_content_to_chat_message(content, conversation_id)

        if param is None:
            return param

        if not isinstance(content, conversation.AssistantContent):
            return param

        if not content.tool_calls:
            return param

        if conversation_id is None:
            param.pop("thinking_content", None)
            return param

        all_signed = all(
            self._get_thought_signature(conversation_id, tc.id) is not None
            for tc in content.tool_calls
        )

        if not all_signed:
            param.pop("thinking_content", None)
            return param

        if content.thinking_content:
            param["thinking_content"] = content.thinking_content

        for i, tc in enumerate(content.tool_calls):
            sig = self._get_thought_signature(conversation_id, tc.id)
            if sig:
                param["tool_calls"][i]["extra_content"] = {
                    "google": {"thought_signature": sig}
                }

        return param


class GoogleGeminiConversationEntity(GoogleGeminiMixin, LocalAiConversationEntity):
    """Conversation agent for Google Gemini servers."""


class GoogleGeminiAITaskEntity(GoogleGeminiMixin, LocalAITaskEntity):
    """AI Task entity for Google Gemini servers."""
