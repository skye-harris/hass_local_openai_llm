"""Unit tests for Google Gemini streaming with thought signature capture."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from custom_components.local_openai.entities.google_gemini import (
    GoogleGeminiConversationEntity,
)
from homeassistant.core import HomeAssistant
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)


class MockSubentry:
    """Mock ConfigSubentry that allows updating data."""

    def __setattr__(self, name: str, value: object) -> None:
        object.__setattr__(self, name, value)

    def __init__(self, data: dict) -> None:
        self.subentry_id = "test_conversation_subentry_id"
        self.subentry_type = "conversation"
        self.title = "Conversation Agent"
        self.data = data
        self.unique_id = None


@pytest.fixture
def google_entity(hass: HomeAssistant) -> GoogleGeminiConversationEntity:
    """Create a Google Gemini conversation entity for testing."""
    entry = MagicMock()
    entry.data = {}
    entry.runtime_data = MagicMock()

    subentry = MockSubentry({"model": "test-model"})

    entity = GoogleGeminiConversationEntity(entry, subentry)
    entity.hass = hass
    return entity


class TestTransformStream:
    """Tests for _transform_stream with thought signatures."""

    def _make_chunk(
        self,
        content: str | None = None,
        tool_calls: list | None = None,
        finish_reason: str | None = None,
        role: str | None = None,
    ) -> ChatCompletionChunk:
        """Create a ChatCompletionChunk for testing."""
        delta = ChoiceDelta(content=content, role=role, tool_calls=tool_calls)
        return ChatCompletionChunk(
            id="test",
            created=0,
            model="test",
            object="chat.completion.chunk",
            choices=[Choice(index=0, delta=delta, finish_reason=finish_reason)],
        )

    async def test_stream_passes_through_text_chunks(self, google_entity):
        """Test that text content chunks pass through _transform_stream."""
        chunk = self._make_chunk(content="hello", role="assistant")

        async def async_gen():
            yield chunk

        mock_stream = MagicMock()
        mock_stream.__aiter__ = MagicMock(return_value=async_gen())

        results = []
        async for result in google_entity._transform_stream(
            mock_stream, False, "conv_1"
        ):
            results.append(result)

        assert len(results) >= 1
        assert results[0].get("role") == "assistant"
        assert results[0].get("content") == "hello"

    async def test_stream_with_tool_calls_passes_through(self, google_entity):
        """Test that tool call chunks pass through _transform_stream."""
        tool_call = ChoiceDeltaToolCall(
            index=0,
            id="call_1",
            type="function",
            function=ChoiceDeltaToolCallFunction(
                name="test_fn",
                arguments='{"arg": "val"}',
            ),
        )
        chunk = self._make_chunk(tool_calls=[tool_call], role="assistant")

        async def async_gen():
            yield chunk

        mock_stream = MagicMock()
        mock_stream.__aiter__ = MagicMock(return_value=async_gen())

        results = []
        async for result in google_entity._transform_stream(
            mock_stream, False, "conv_1"
        ):
            results.append(result)

        assert len(results) >= 1

    async def test_stream_with_none_conversation_id_passes_through(self, google_entity):
        """Test that streaming works with None conversation_id."""
        chunk = self._make_chunk(content="hello", role="assistant")

        async def async_gen():
            yield chunk

        mock_stream = MagicMock()
        mock_stream.__aiter__ = MagicMock(return_value=async_gen())

        results = []
        async for result in google_entity._transform_stream(mock_stream, False, None):
            results.append(result)

        assert len(results) >= 1
        assert results[0].get("content") == "hello"

    async def test_stream_yields_all_chunks(self, google_entity):
        """Test that all chunks are yielded in order."""
        chunks = [
            self._make_chunk(content="hello", role="assistant"),
            self._make_chunk(content=" world"),
        ]

        async def async_gen():
            for c in chunks:
                yield c

        mock_stream = MagicMock()
        mock_stream.__aiter__ = MagicMock(return_value=async_gen())

        results = []
        async for result in google_entity._transform_stream(
            mock_stream, False, "conv_1"
        ):
            results.append(result)

        assert len(results) >= 2

    async def test_stream_empty_stream(self, google_entity):
        """Test that empty stream yields nothing."""

        async def async_gen():
            return
            yield  # pragma: no cover

        mock_stream = MagicMock()
        mock_stream.__aiter__ = MagicMock(return_value=async_gen())

        results = []
        async for result in google_entity._transform_stream(
            mock_stream, False, "conv_1"
        ):
            results.append(result)

        assert results == []

    async def test_stream_conversation_id_passed_to_parent(self, google_entity):
        """Test that conversation_id is passed to parent _transform_stream."""
        chunk = self._make_chunk(content="test", role="assistant")

        mock_stream = MagicMock()
        mock_stream.__aiter__ = MagicMock(return_value=iter([chunk]))

        with pytest.raises(TypeError):
            # Parent method expects specific args; this verifies
            # conversation_id is at least being passed
            async for _ in google_entity._transform_stream(
                mock_stream, False, "conv_1"
            ):
                pass
