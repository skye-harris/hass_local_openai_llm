"""Tests for converting chat log content to OpenAI chat messages."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from homeassistant.components import conversation
from homeassistant.const import CONF_MODEL
from homeassistant.core import HomeAssistant

from custom_components.local_openai import conversation as local_conversation

ConversationEntity = local_conversation.LocalAiConversationEntity


@pytest.fixture
def entity(hass: HomeAssistant) -> ConversationEntity:
    """Create a conversation entity for conversion tests."""
    entry = MagicMock()
    entry.data = {}
    entry.runtime_data = MagicMock()

    subentry = MagicMock()
    subentry.data = {CONF_MODEL: "test-model"}

    entity = ConversationEntity(entry, subentry)
    entity.hass = hass
    return entity


def _image_attachment(tmp_path: Path) -> conversation.Attachment:
    """Create an image attachment backed by a real file."""
    image_path = tmp_path / "snapshot.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nimage-bytes")
    return conversation.Attachment(
        media_content_id="media-source://camera/camera.front_door",
        mime_type="image/png",
        path=image_path,
    )


async def test_image_only_user_content_keeps_attachment(
    entity: ConversationEntity,
    tmp_path: Path,
) -> None:
    """A user message with empty text still sends its image."""
    message = await entity._convert_content_to_chat_message(
        conversation.UserContent(
            content="",
            attachments=[_image_attachment(tmp_path)],
        )
    )

    assert message is not None
    assert message["role"] == "user"

    parts = message["content"]
    assert isinstance(parts, list)
    assert len(parts) == 1
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,")


async def test_text_and_image_user_content_keeps_both(
    entity: ConversationEntity,
    tmp_path: Path,
) -> None:
    """Text and image parts are both sent when text is present."""
    message = await entity._convert_content_to_chat_message(
        conversation.UserContent(
            content="What is in this image?",
            attachments=[_image_attachment(tmp_path)],
        )
    )

    assert message is not None

    parts = message["content"]
    assert isinstance(parts, list)
    assert [part["type"] for part in parts] == ["image_url", "text"]
    assert parts[1]["text"] == "What is in this image?"


async def test_empty_user_content_without_attachments_is_skipped(
    entity: ConversationEntity,
) -> None:
    """A user message with no text and no attachments is still dropped."""
    message = await entity._convert_content_to_chat_message(
        conversation.UserContent(content="")
    )

    assert message is None
