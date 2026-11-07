"""
LFM model-specific entity support.

LFM 2.x models emit tool calls as text instead of using the OpenAI ``tools``
delta format::

    <|tool_call_start|>[HassTurnOn(name='TV LED', area='Living Room')]<|tool_call_end|>

This module adapts the generic entity for that dialect: the tool schemas travel
in ``chat_template_kwargs`` plus a system-message listing, and the streaming
transform turns the bracketed text back into ``llm.ToolInput`` objects.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import TYPE_CHECKING, Any

from homeassistant.components import conversation
from homeassistant.helpers import llm
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from openai.types.chat import ChatCompletionChunk, ChatCompletionFunctionToolParam
    from openai.types.chat.chat_completion_chunk import AsyncStream

    from custom_components.local_openai.entity import AgentContext, GenerationMetadata

_LOGGER = logging.getLogger(__name__)

TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"

# Longest marker prefix that has to be held back until it is known whether it
# starts a tool call. One character less than the marker itself.
_HOLDBACK_CHARS = max(len(TOOL_CALL_START), len(TOOL_CALL_END)) - 1


def _parse_lfm_tool_calls(
    content: str, allowed_tools: set[str] | None = None
) -> list[dict]:
    """
    Parse LFM-style tool calls out of a content string.

    Input examples:
    - "[GetLiveContext()]"
    - "[HassTurnOn(name='TV LED', area='Living Room')]"
    - "[func1(arg1='val'), func2(arg2=123)]"

    Parsed with ``ast`` rather than a regex so quoted arguments, nested
    structures and multiple calls per bracket all survive intact.
    """
    if not content:
        return []

    # Extract content inside brackets if enclosed in []
    match = re.search(r"\[(.*)\]", content, re.DOTALL)
    expr_str = f"[{match.group(1).strip()}]" if match else content.strip()

    try:
        tree = ast.parse(expr_str, mode="eval")
    except SyntaxError as err:
        _LOGGER.warning(
            "Failed to parse LFM tool call expression '%s': %s", expr_str, err
        )
        return []

    nodes: list[ast.AST] = []
    if isinstance(tree.body, ast.List):
        nodes = list(tree.body.elts)
    elif isinstance(tree.body, ast.Call):
        nodes = [tree.body]

    calls: list[dict] = []
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue

        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        if not func_name:
            continue

        if allowed_tools is not None and func_name not in allowed_tools:
            _LOGGER.warning(
                "Attempted to call unknown function: %s (allowed: %s)",
                func_name,
                allowed_tools,
            )
            continue

        kwargs: dict[str, Any] = {}
        for keyword in node.keywords:
            if keyword.arg:
                try:
                    kwargs[keyword.arg] = ast.literal_eval(keyword.value)
                except (ValueError, SyntaxError):
                    kwargs[keyword.arg] = str(keyword.value)

        calls.append({"name": func_name, "args": kwargs})

    return calls


def _format_lfm_tool_call_string(tool_calls: list[llm.ToolInput]) -> str:
    """Format ToolInput objects into LFM's tool call text representation."""
    calls_str = []
    for call in tool_calls:
        args_formatted = [f"{key}={value!r}" for key, value in call.tool_args.items()]
        calls_str.append(f"{call.tool_name}({', '.join(args_formatted)})")
    return f"{TOOL_CALL_START}[{', '.join(calls_str)}]{TOOL_CALL_END}"


class LfmMixin:
    """Mixin adding LFM tool-calling support to the generic entities."""

    def _customize_model_args(
        self,
        model_args: dict[str, Any],
        tools: list[ChatCompletionFunctionToolParam] | None,
        chat_log: conversation.ChatLog,
    ) -> None:
        """
        Move tool schemas out of the OpenAI fields and into the prompt.

        LFM's chat template expects the schemas in ``chat_template_kwargs``, and
        servers must not attempt their own tool calling on top.
        """
        if not tools:
            return

        model_args.pop("tools", None)
        model_args.pop("tool_choice", None)

        extra_body = model_args.setdefault("extra_body", {})
        chat_template_kwargs = extra_body.setdefault("chat_template_kwargs", {})
        chat_template_kwargs["tools"] = tools
        extra_body["function_call"] = "none"

        # Remember the schemas so they survive the per-iteration rebuild of
        # extra_body done by _get_extra_body_args.
        self._lfm_tools = tools

        tools_json = [tool["function"] for tool in tools if "function" in tool]
        tools_prompt = f"\n\nList of tools: {json.dumps(tools_json)}"

        messages = model_args.get("messages", [])
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] += tools_prompt
        elif messages:
            messages.insert(
                0,
                {"role": "system", "content": f"List of tools: {json.dumps(tools_json)}"},
            )

    def _get_extra_body_args(self, context: AgentContext) -> None:
        """Re-apply the LFM tool schemas after the base class rebuilt extra_body."""
        super()._get_extra_body_args(context)

        tools = getattr(self, "_lfm_tools", None)
        if not tools:
            return

        extra_body = context.extra_body
        extra_body.setdefault("chat_template_kwargs", {})["tools"] = tools
        extra_body["function_call"] = "none"

    async def _convert_content_to_chat_message(
        self,
        content: conversation.Content,
        conversation_id: str | None = None,
    ) -> ChatCompletionMessageParam | None:
        """Render assistant tool calls in LFM's text dialect."""
        if isinstance(content, conversation.AssistantContent) and content.tool_calls:
            lfm_tool_str = _format_lfm_tool_call_string(content.tool_calls)
            text_content = content.content or ""
            full_content = (
                f"{lfm_tool_str} {text_content}".strip()
                if text_content
                else lfm_tool_str
            )
            return ChatCompletionAssistantMessageParam(
                role="assistant",
                content=full_content,
            )

        return await super()._convert_content_to_chat_message(
            content, conversation_id
        )

    async def _transform_stream(
        self,
        stream: AsyncStream[ChatCompletionChunk],
        strip_emojis: bool,
        conversation_id: str | None = None,
        meta: GenerationMetadata | None = None,
    ) -> AsyncGenerator[conversation.AssistantContentDeltaDict, None]:
        """
        Turn bracketed LFM tool-call text back into tool calls.

        Wraps the base transform instead of reimplementing it, so thinking
        blocks, role deltas, timestamp handling and the usage/timings metadata
        keep behaving exactly as they do for every other server type. Only the
        content stream is inspected: text is emitted as it arrives except for a
        short holdback that may still turn out to be a tool-call marker.
        """
        buffer = ""
        in_call = False
        pending_call = ""

        async for delta in super()._transform_stream(
            stream=stream,
            strip_emojis=strip_emojis,
            conversation_id=conversation_id,
            meta=meta,
        ):
            content = delta.get("content")
            if content is None:
                yield delta
                continue

            chunk = {key: value for key, value in delta.items() if key != "content"}
            buffer += content
            visible = ""

            while True:
                if not in_call:
                    index = buffer.find(TOOL_CALL_START)
                    if index == -1:
                        # Keep a possible partial marker at the tail.
                        if len(buffer) > _HOLDBACK_CHARS:
                            visible += buffer[:-_HOLDBACK_CHARS]
                            buffer = buffer[-_HOLDBACK_CHARS:]
                        break
                    visible += buffer[:index]
                    buffer = buffer[index + len(TOOL_CALL_START) :]
                    in_call = True
                    pending_call = ""
                else:
                    index = buffer.find(TOOL_CALL_END)
                    if index == -1:
                        pending_call += buffer
                        buffer = ""
                        break
                    pending_call += buffer[:index]
                    buffer = buffer[index + len(TOOL_CALL_END) :]
                    in_call = False

                    calls = _parse_lfm_tool_calls(pending_call)
                    if calls:
                        _LOGGER.debug("LFM tool calls: %s", calls)
                        chunk["tool_calls"] = [
                            llm.ToolInput(
                                id=f"call_{position}",
                                tool_name=call["name"],
                                tool_args=call["args"],
                            )
                            for position, call in enumerate(calls)
                        ]
                    pending_call = ""

            if visible:
                chunk["content"] = visible
            if chunk:
                yield chunk

        if in_call:
            _LOGGER.warning(
                "Unterminated LFM tool call, dropping: %s", pending_call + buffer
            )
        elif buffer:
            yield {"content": buffer}


def get_conversation_config_schema() -> dict:
    """Return the conversation config schema for LFM."""
    return {}


def get_ai_task_config_schema() -> dict:
    """Return the AI task config schema for LFM."""
    return get_conversation_config_schema()


# Imported at the bottom to avoid a circular import: the entity modules import
# the mixin's server type from const, and these bases import conversation.py.
from custom_components.local_openai.ai_task import LocalAITaskEntity  # noqa: E402
from custom_components.local_openai.conversation import (  # noqa: E402
    LocalAiConversationEntity,
)


class LfmConversationEntity(LfmMixin, LocalAiConversationEntity):
    """Conversation agent with LFM tool calling support."""


class LfmAITaskEntity(LfmMixin, LocalAITaskEntity):
    """AI Task entity with LFM tool calling support."""
