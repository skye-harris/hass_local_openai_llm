"""Tests for LocalAiEntity._run_agent_loop."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest
from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from custom_components.local_openai.entity import MAX_TOOL_ITERATIONS
from custom_components.local_openai.conversation import LocalAiConversationEntity


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
        chat_log.content = []

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
        chat_log.content = []

        def get_unresponded():
            return chat_log.content and chat_log.content[-1].role == "tool_result"

        chat_log.unresponded_tool_results = property(get_unresponded)

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

        with pytest.raises(
            HomeAssistantError,
            match="API server returned an error. Check the system logs for further details.",
        ):
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

        with pytest.raises(
            HomeAssistantError,
            match="Error handling API response. Check the system logs for further details.",
        ):
            await entity._run_agent_loop(
                entity.entry.runtime_data,
                model_args,
                chat_log,
                strip_emojis=False,
            )

    async def test_tool_args_decode_failure_injects_error_result(
        self,
        hass: HomeAssistant,
        mock_conversation_entity: conversation.ConversationAgent,
    ):
        """Test malformed tool call arguments injects error result and continues loop."""
        entity = mock_conversation_entity

        tool_call = ChoiceDeltaToolCall(
            index=0,
            id="call_1",
            type="function",
            function=ChoiceDeltaToolCallFunction(
                name="test_fn",
                arguments='{"arg": broken-json',
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
            usage=None,
        )

        # Add finish_reason to trigger the tool call parsing
        chunk_with_finish = ChatCompletionChunk(
            id="test",
            created=0,
            model="test",
            object="chat.completion.chunk",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(),
                    finish_reason="tool_calls",
                )
            ],
            usage=None,
        )

        mock_stream = MagicMock()

        async def async_gen():
            for item in [chunk, chunk_with_finish]:
                yield item

        mock_stream.__aiter__ = MagicMock(return_value=async_gen())
        entity.entry.runtime_data.chat.completions.create = AsyncMock(
            return_value=mock_stream
        )

        async def stream_content(
            entity_id, stream_gen
        ) -> AsyncGenerator[
            conversation.AssistantContent | conversation.ToolResultContent, None
        ]:
            assistant = MockContent(
                content="",
                tool_calls=[
                    llm.ToolInput(
                        id="call_1",
                        tool_name="test_fn",
                        tool_args={"json_decode_failure": '{"arg": broken-json'},
                        external=True,
                    )
                ],
            )
            yield assistant
            chat_log.content.append(assistant)
            error_content = conversation.ToolResultContent(
                agent_id="test_agent",
                tool_call_id="call_1",
                tool_name="test_fn",
                tool_result={
                    "error": 'Failed to parse tool arguments: {"arg": broken-json\n\nCheck your JSON and try again.',
                },
            )
            yield error_content
            chat_log.content.append(error_content)

        chat_log = MagicMock(spec=conversation.ChatLog)
        chat_log.async_add_delta_content_stream = stream_content
        chat_log.content = []

        def get_unresponded():
            return chat_log.content and chat_log.content[-1].role == "tool_result"

        chat_log.unresponded_tool_results = property(get_unresponded)

        call_count = 0

        def create_side_effect(**kwargs) -> MagicMock:
            nonlocal call_count
            call_count += 1
            mock_stream = MagicMock()
            mock_stream.__aiter__ = MagicMock(return_value=async_gen())
            return mock_stream

        entity.entry.runtime_data.chat.completions.create = AsyncMock(
            side_effect=create_side_effect
        )

        model_args = {"model": "test", "messages": []}

        # Should not raise - error is injected and loop continues
        await entity._run_agent_loop(
            entity.entry.runtime_data,
            model_args,
            chat_log,
            strip_emojis=False,
        )

        # Verify the loop ran all iterations (tool error keeps it going)
        assert call_count == MAX_TOOL_ITERATIONS
        # Verify tool error result was injected into chat log in each iteration
        assert len(chat_log.content) == MAX_TOOL_ITERATIONS * 2
        for i in range(0, len(chat_log.content), 2):
            assistant_content = chat_log.content[i]
            assert isinstance(assistant_content, conversation.AssistantContent)
            assert assistant_content.tool_calls is not None
            assert len(assistant_content.tool_calls) == 1
            assert assistant_content.tool_calls[0].external is True
            error_content = chat_log.content[i + 1]
            assert isinstance(error_content, conversation.ToolResultContent)
            assert error_content.tool_name == "test_fn"
            assert error_content.tool_call_id == "call_1"
            assert (
                "Failed to parse tool arguments" in error_content.tool_result["error"]
            )

    async def test_stop_directive_injected_on_final_iteration(
        self,
        hass: HomeAssistant,
        mock_conversation_entity: conversation.ConversationAgent,
    ):
        """Test stop directive is injected and tool_choice set to none on the final iteration."""
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

        chunk_with_finish = ChatCompletionChunk(
            id="test",
            created=0,
            model="test",
            object="chat.completion.chunk",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(),
                    finish_reason="tool_calls",
                )
            ],
            usage=None,
        )

        call_count = 0
        captured_kwargs: list[dict] = []

        def create_side_effect(**kwargs) -> MagicMock:
            nonlocal call_count
            call_count += 1
            captured_kwargs.append(deepcopy(kwargs))
            if call_count == MAX_TOOL_ITERATIONS:
                last_msg = kwargs["messages"][-1]
                assert "maximum number of tool call iterations" in last_msg.get(
                    "content", ""
                ), (
                    f"Iteration {call_count}: last_msg content = {last_msg.get('content')!r}"
                )
            mock_stream = MagicMock()
            mock_stream.__aiter__ = MagicMock(
                return_value=iter([chunk, chunk_with_finish])
            )
            return mock_stream

        entity.entry.runtime_data.chat.completions.create = AsyncMock(
            side_effect=create_side_effect
        )

        async def stream_content(
            entity_id, stream_gen
        ) -> AsyncGenerator[
            conversation.AssistantContent | conversation.ToolResultContent, None
        ]:
            assistant = MockContent(
                content="",
                tool_calls=[
                    llm.ToolInput(
                        id="call_1",
                        tool_name="test_fn",
                        tool_args={"arg": "val"},
                        external=True,
                    )
                ],
            )
            yield assistant
            chat_log.content.append(assistant)
            error_content = conversation.ToolResultContent(
                agent_id="test_agent",
                tool_call_id="call_1",
                tool_name="test_fn",
                tool_result={"error": "tool failed"},
            )
            yield error_content
            chat_log.content.append(error_content)

        chat_log = MagicMock(spec=conversation.ChatLog)
        chat_log.async_add_delta_content_stream = stream_content
        chat_log.content = []

        def get_unresponded():
            return chat_log.content and chat_log.content[-1].role == "tool_result"

        chat_log.unresponded_tool_results = property(get_unresponded)

        model_args = {"model": "test", "messages": []}

        await entity._run_agent_loop(
            entity.entry.runtime_data,
            model_args,
            chat_log,
            strip_emojis=False,
        )

        assert call_count == MAX_TOOL_ITERATIONS

        last_call = captured_kwargs[-1]
        assert call_count == MAX_TOOL_ITERATIONS, (
            f"Expected {MAX_TOOL_ITERATIONS} calls, got {call_count}"
        )
        assert last_call["tool_choice"] == "none"

        last_message = last_call["messages"][-1]
        assert last_message["role"] == "tool"
        directive = (
            "\n\nYou have reached the maximum number of tool call iterations. "
            "You must now provide your final response directly to the user "
            "without calling any more tools."
        )
        assert last_message["content"] == '{"error": "tool failed"}' + directive, (
            f"Got: {last_message['content']!r}"
        )

        for i in range(MAX_TOOL_ITERATIONS - 1):
            assert captured_kwargs[i].get("tool_choice") != "none"


class TestInjectStopDirective:
    """Test _inject_stop_directive static method."""

    def test_empty_messages(self):
        """Test empty message list is returned unchanged."""
        messages: list[dict] = []
        result = LocalAiConversationEntity._inject_stop_directive(messages)
        assert result == []

    def test_no_last_tool_message(self):
        """Test when last message is not a tool message, list is returned unchanged."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        result = LocalAiConversationEntity._inject_stop_directive(messages)
        assert result == messages

    def test_with_empty_tool_content(self):
        """Test directive appended to empty tool content."""
        messages = [
            {"role": "tool", "tool_call_id": "1", "tool_name": "fn", "content": ""}
        ]
        result = LocalAiConversationEntity._inject_stop_directive(messages)
        directive = (
            "\n\nYou have reached the maximum number of tool call iterations. "
            "You must now provide your final response directly to the user "
            "without calling any more tools."
        )
        assert result[0]["content"] == directive

    def test_with_existing_tool_content(self):
        """Test directive appended after existing tool content."""
        messages = [
            {
                "role": "tool",
                "tool_call_id": "1",
                "tool_name": "fn",
                "content": "Previous result",
            }
        ]
        result = LocalAiConversationEntity._inject_stop_directive(messages)
        directive = (
            "\n\nYou have reached the maximum number of tool call iterations. "
            "You must now provide your final response directly to the user "
            "without calling any more tools."
        )
        assert result[0]["content"] == "Previous result" + directive
