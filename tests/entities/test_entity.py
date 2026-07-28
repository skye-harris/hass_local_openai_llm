"""Tests for LocalAiEntity._run_agent_loop."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest
from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from custom_components.local_openai.entity import MAX_TOOL_ITERATIONS


class MockContent(conversation.AssistantContent):
    """Minimal Content for async_add_delta_content_stream."""

    def __init__(
        self,
        content: str = "hello",
        thinking_content: str | None = None,
        tool_calls: list | None = None,
    ) -> None:
        super().__init__(
            agent_id="test_agent",
            content=content,
            thinking_content=thinking_content,
            tool_calls=tool_calls,
        )


class TestRunAgentLoopSingleIteration:
    """Test _run_agent_loop with a single successful iteration."""

    async def test_single_iteration_no_tools(
        self,
        hass: HomeAssistant,
        mock_conversation_entity: conversation.ConversationAgent,
    ):
        """Test loop completes in one iteration when no tool results."""
        entity = mock_conversation_entity

        chunk = ChatCompletionChunk(
            id="test",
            created=0,
            model="test",
            object="chat.completion.chunk",
            choices=[
                Choice(index=0, delta=ChoiceDelta(content="hi", role="assistant"))
            ],
        )

        mock_stream = MagicMock()
        mock_stream.__aiter__ = MagicMock(return_value=iter([chunk]))
        entity.entry.runtime_data.chat.completions.create = AsyncMock(
            return_value=mock_stream
        )

        async def stream_content(
            entity_id, stream_gen
        ) -> AsyncGenerator[MockContent, None]:
            yield MockContent(content="hi")

        chat_log = MagicMock(spec=conversation.ChatLog)
        chat_log.async_add_delta_content_stream = stream_content
        chat_log.unresponded_tool_results = []

        model_args = {"model": "test", "messages": []}

        await entity._run_agent_loop(
            entity.entry.runtime_data,
            model_args,
            chat_log,
            strip_emojis=False,
        )

        entity.entry.runtime_data.chat.completions.create.assert_called_once()
        assert len(model_args["messages"]) == 1

    async def test_tool_call_iteration(
        self,
        hass: HomeAssistant,
        mock_conversation_entity: conversation.ConversationAgent,
    ):
        """Test loop continues when there are unresponded tool results."""
        entity = mock_conversation_entity

        tool_call = ChoiceDeltaToolCall(
            index=0,
            id="call_1",
            type="function",
            function=ChoiceDeltaToolCallFunction(
                name="test_fn",
                arguments='{"arg": "val"}',
            ),
        )
        chunk = ChatCompletionChunk(
            id="test",
            created=0,
            model="test",
            object="chat.completion.chunk",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        role="assistant",
                        tool_calls=[tool_call],
                    ),
                )
            ],
        )

        call_count = 0

        def create_side_effect(**kwargs) -> MagicMock:
            nonlocal call_count
            call_count += 1
            mock_stream = MagicMock()
            mock_stream.__aiter__ = MagicMock(return_value=iter([chunk]))
            return mock_stream

        entity.entry.runtime_data.chat.completions.create = AsyncMock(
            side_effect=create_side_effect
        )

        async def stream_content(
            entity_id, stream_gen
        ) -> AsyncGenerator[MockContent, None]:
            yield MockContent(content="")

        chat_log = MagicMock(spec=conversation.ChatLog)
        chat_log.async_add_delta_content_stream = stream_content
        chat_log.unresponded_tool_results = [MagicMock()]

        model_args = {"model": "test", "messages": []}

        await entity._run_agent_loop(
            entity.entry.runtime_data,
            model_args,
            chat_log,
            strip_emojis=False,
        )

        assert call_count == MAX_TOOL_ITERATIONS
        assert len(model_args["messages"]) == MAX_TOOL_ITERATIONS


class TestRunAgentLoopErrorHandling:
    """Test _run_agent_loop error handling."""

    async def test_openai_error_raises_homeassistant_error(
        self,
        hass: HomeAssistant,
        mock_conversation_entity: conversation.ConversationAgent,
    ):
        """Test OpenAIError is wrapped in HomeAssistantError."""
        entity = mock_conversation_entity

        entity.entry.runtime_data.chat.completions.create = AsyncMock(
            side_effect=openai.APIConnectionError(request=MagicMock())
        )

        chat_log = MagicMock(spec=conversation.ChatLog)
        chat_log.unresponded_tool_results = []

        model_args = {"model": "test", "messages": []}

        with pytest.raises(HomeAssistantError, match="Error talking to API"):
            await entity._run_agent_loop(
                entity.entry.runtime_data,
                model_args,
                chat_log,
                strip_emojis=False,
            )

    async def test_stream_error_raises_homeassistant_error(
        self,
        hass: HomeAssistant,
        mock_conversation_entity: conversation.ConversationAgent,
    ):
        """Test streaming error is wrapped in HomeAssistantError."""
        entity = mock_conversation_entity

        chunk = ChatCompletionChunk(
            id="test",
            created=0,
            model="test",
            object="chat.completion.chunk",
            choices=[
                Choice(index=0, delta=ChoiceDelta(content="hi", role="assistant"))
            ],
        )

        mock_stream = MagicMock()
        mock_stream.__aiter__ = MagicMock(return_value=iter([chunk]))
        entity.entry.runtime_data.chat.completions.create = AsyncMock(
            return_value=mock_stream
        )

        async def stream_content_fail(
            entity_id, stream_gen
        ) -> AsyncGenerator[MockContent, None]:
            raise RuntimeError("stream broken")
            yield  # pragma: no cover

        chat_log = MagicMock(spec=conversation.ChatLog)
        chat_log.async_add_delta_content_stream = stream_content_fail
        chat_log.unresponded_tool_results = []

        model_args = {"model": "test", "messages": []}

        with pytest.raises(HomeAssistantError, match="Error handling API response"):
            await entity._run_agent_loop(
                entity.entry.runtime_data,
                model_args,
                chat_log,
                strip_emojis=False,
            )
