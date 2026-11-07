"""Tests for the LFM tool-calling dialect.

Covers the three pieces that make LFM work with this integration: parsing the
bracketed text format, moving the tool schemas into the prompt, and turning the
streamed text back into tool calls without leaking the markers to the user.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from custom_components.local_openai.entities.lfm import (
    LfmConversationEntity,
    LfmMixin,
    _format_lfm_tool_call_string,
    _parse_lfm_tool_calls,
)
from custom_components.local_openai.entity import AgentContext, GenerationMetadata
from homeassistant import config_entries
from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "HassTurnOn",
            "description": "Turn on a device",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
        },
    }
]


class MockSubentry:
    """Minimal ConfigSubentry stand-in."""

    subentry_id = "test_conversation_subentry_id"
    subentry_type = "conversation"
    title = "Conversation Agent"
    unique_id = None

    def __init__(self, data: dict) -> None:
        self.data = data


@pytest.fixture
def lfm_entity(hass: HomeAssistant) -> LfmConversationEntity:
    """Create an LFM conversation entity for testing."""
    entry = MagicMock(spec=config_entries.ConfigEntry)
    entry.data = {}
    entry.runtime_data = MagicMock()

    entity = LfmConversationEntity(entry, MockSubentry({"model": "lfm"}))
    entity.hass = hass
    return entity


def _chunk(
    content: str | None = None,
    role: str | None = None,
    finish_reason: str | None = None,
) -> ChatCompletionChunk:
    """Build a chunk the way an OpenAI-compatible server would send it."""
    delta = ChoiceDelta(content=content, role=role)
    return ChatCompletionChunk(
        id="test",
        created=0,
        model="lfm",
        object="chat.completion.chunk",
        choices=[Choice(index=0, delta=delta, finish_reason=finish_reason)],
    )


async def _collect(
    entity: LfmConversationEntity,
    chunks: list[ChatCompletionChunk],
    meta: GenerationMetadata | None = None,
) -> list[dict]:
    """Run chunks through _transform_stream and return the yielded deltas."""

    async def async_gen():
        for chunk in chunks:
            yield chunk

    stream = MagicMock()
    stream.__aiter__ = MagicMock(return_value=async_gen())
    return [
        delta
        async for delta in entity._transform_stream(
            stream, False, "conv_1", meta or GenerationMetadata()
        )
    ]


class TestParseLfmToolCalls:
    """Parsing of the bracketed tool-call text."""

    def test_single_call_without_args(self) -> None:
        calls = _parse_lfm_tool_calls("[GetLiveContext()]", {"GetLiveContext"})
        assert calls == [{"name": "GetLiveContext", "args": {}}]

    def test_single_call_with_quoted_args(self) -> None:
        calls = _parse_lfm_tool_calls(
            "[HassTurnOn(name='TV LED', area='Living Room')]", {"HassTurnOn"}
        )
        assert calls == [
            {
                "name": "HassTurnOn",
                "args": {"name": "TV LED", "area": "Living Room"},
            }
        ]

    def test_multiple_calls(self) -> None:
        calls = _parse_lfm_tool_calls(
            '[update_status(id="1"), get_info(id="2")]',
            {"update_status", "get_info"},
        )
        assert [call["name"] for call in calls] == ["update_status", "get_info"]
        assert calls[1]["args"] == {"id": "2"}

    def test_nested_and_numeric_args(self) -> None:
        calls = _parse_lfm_tool_calls(
            "[set_timer(seconds=90, tags=['a', 'b'])]", {"set_timer"}
        )
        assert calls[0]["args"] == {"seconds": 90, "tags": ["a", "b"]}

    def test_unknown_function_is_dropped(self) -> None:
        calls = _parse_lfm_tool_calls(
            '[allowed_func(arg="1"), disallowed_func(arg="2")]', {"allowed_func"}
        )
        assert [call["name"] for call in calls] == ["allowed_func"]

    def test_empty_content(self) -> None:
        assert _parse_lfm_tool_calls("", {"func"}) == []

    def test_unparsable_content(self) -> None:
        assert _parse_lfm_tool_calls("[not valid python(", {"func"}) == []


class TestFormatLfmToolCallString:
    """Rendering previous tool calls back into the transcript."""

    def test_round_trip_shape(self) -> None:
        tool_input = llm.ToolInput(
            id="call_0",
            tool_name="HassTurnOn",
            tool_args={"name": "TV LED", "area": "Living Room"},
        )
        assert _format_lfm_tool_call_string([tool_input]) == (
            "<|tool_call_start|>"
            "[HassTurnOn(name='TV LED', area='Living Room')]"
            "<|tool_call_end|>"
        )


class TestCustomizeModelArgs:
    """Tool schemas belong in the prompt, not in the OpenAI tools field."""

    def test_tools_move_into_chat_template_kwargs(
        self, lfm_entity: LfmConversationEntity
    ) -> None:
        model_args: dict[str, Any] = {
            "model": "lfm",
            "tools": TOOLS,
            "tool_choice": "auto",
            "messages": [{"role": "system", "content": "You are helpful."}],
        }

        lfm_entity._customize_model_args(model_args, TOOLS, MagicMock())

        assert "tools" not in model_args
        assert "tool_choice" not in model_args
        assert model_args["extra_body"]["chat_template_kwargs"]["tools"] == TOOLS
        assert model_args["extra_body"]["function_call"] == "none"
        assert "List of tools:" in model_args["messages"][0]["content"]
        assert "HassTurnOn" in model_args["messages"][0]["content"]

    def test_no_tools_leaves_model_args_alone(
        self, lfm_entity: LfmConversationEntity
    ) -> None:
        model_args: dict[str, Any] = {
            "model": "lfm",
            "messages": [{"role": "system", "content": "hi"}],
        }

        lfm_entity._customize_model_args(model_args, None, MagicMock())

        assert model_args == {
            "model": "lfm",
            "messages": [{"role": "system", "content": "hi"}],
        }


class _RebuildingBase:
    """Stand-in for the generic entity: rebuilds extra_body every iteration."""

    def _get_extra_body_args(self, context: AgentContext) -> None:
        context.extra_body.clear()
        context.extra_body["stream_options"] = {"include_usage": True}


class _EntityWithRebuild(LfmMixin, _RebuildingBase):
    """LFM mixin on top of a base that discards extra_body each iteration."""


class TestExtraBodyPerIteration:
    """The schemas must survive the per-iteration rebuild of extra_body."""

    def test_tools_are_reapplied(self) -> None:
        entity = _EntityWithRebuild()
        entity._lfm_tools = TOOLS
        context = AgentContext(conversation_id="conv_1", iteration=1)

        entity._get_extra_body_args(context)

        assert context.extra_body["chat_template_kwargs"]["tools"] == TOOLS
        assert context.extra_body["function_call"] == "none"
        assert context.extra_body["stream_options"] == {"include_usage": True}

    def test_no_tools_injects_nothing(self) -> None:
        context = AgentContext(conversation_id="conv_1")

        _EntityWithRebuild()._get_extra_body_args(context)

        assert "chat_template_kwargs" not in context.extra_body
        assert "function_call" not in context.extra_body


class TestConvertContentToChatMessage:
    """Assistant tool calls are replayed to the model in LFM's dialect."""

    async def test_tool_calls_become_text(
        self, lfm_entity: LfmConversationEntity
    ) -> None:
        content = conversation.AssistantContent(
            agent_id="test_agent",
            content="Turning it on.",
            tool_calls=[
                llm.ToolInput(
                    id="call_0",
                    tool_name="HassTurnOn",
                    tool_args={"name": "TV LED"},
                )
            ],
        )

        message = await lfm_entity._convert_content_to_chat_message(content, "conv_1")

        assert message["role"] == "assistant"
        assert message["content"] == (
            "<|tool_call_start|>[HassTurnOn(name='TV LED')]<|tool_call_end|>"
            " Turning it on."
        )
        assert "tool_calls" not in message

    async def test_plain_text_is_delegated(
        self, lfm_entity: LfmConversationEntity
    ) -> None:
        content = conversation.AssistantContent(agent_id="test_agent", content="Hi")

        message = await lfm_entity._convert_content_to_chat_message(content, "conv_1")

        assert message["role"] == "assistant"
        assert message["content"] == "Hi"


class TestTransformStream:
    """Streaming translation of tool-call text into tool calls."""

    async def test_plain_text_passes_through(
        self, lfm_entity: LfmConversationEntity
    ) -> None:
        deltas = await _collect(
            lfm_entity,
            [_chunk(content="Hello", role="assistant"), _chunk(content=" world")],
        )

        assert "".join(delta.get("content") or "" for delta in deltas) == "Hello world"
        assert not any(delta.get("tool_calls") for delta in deltas)

    async def test_marker_split_across_chunks(
        self, lfm_entity: LfmConversationEntity
    ) -> None:
        deltas = await _collect(
            lfm_entity,
            [
                _chunk(content="Sure <|tool_call_", role="assistant"),
                _chunk(content="start|>[HassTurnOn(name='TV')]<|tool_call_end|>"),
            ],
        )

        # The marker never reaches the user, the prose before it does.
        assert "".join(delta.get("content") or "" for delta in deltas) == "Sure "
        calls = [
            call for delta in deltas for call in (delta.get("tool_calls") or [])
        ]
        assert [call.tool_name for call in calls] == ["HassTurnOn"]
        assert calls[0].tool_args == {"name": "TV"}

    async def test_text_after_call_is_kept(
        self, lfm_entity: LfmConversationEntity
    ) -> None:
        deltas = await _collect(
            lfm_entity,
            [
                _chunk(
                    content="<|tool_call_start|>[HassTurnOff(name='TV')]"
                    "<|tool_call_end|>Done",
                    role="assistant",
                )
            ],
        )

        assert "".join(delta.get("content") or "" for delta in deltas) == "Done"
        calls = [
            call for delta in deltas for call in (delta.get("tool_calls") or [])
        ]
        assert [call.tool_name for call in calls] == ["HassTurnOff"]

    async def test_unterminated_call_is_not_shown(
        self, lfm_entity: LfmConversationEntity
    ) -> None:
        deltas = await _collect(
            lfm_entity,
            [_chunk(content="<|tool_call_start|>[HassTurnOn(name=", role="assistant")],
        )

        assert "".join(delta.get("content") or "" for delta in deltas) == ""
        assert not any(delta.get("tool_calls") for delta in deltas)
