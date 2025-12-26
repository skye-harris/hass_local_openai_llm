"""Base entity for Open Router."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any, Literal

import demoji
import openai
import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_MODEL
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import llm
from homeassistant.helpers.entity import Entity
from openai._streaming import AsyncStream
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_function_tool_call_param import Function
from openai.types.shared_params import FunctionDefinition, ResponseFormatJSONSchema
from openai.types.shared_params.response_format_json_schema import JSONSchema
from voluptuous_openapi import convert

from . import LocalAiConfigEntry
from .const import (
    CONF_MAX_MESSAGE_HISTORY,
    CONF_PARALLEL_TOOL_CALLS,
    CONF_STRIP_EMOJIS,
    CONF_TEMPERATURE,
    CONF_WEAVIATE_API_KEY,
    CONF_WEAVIATE_CLASS_NAME,
    CONF_WEAVIATE_DEFAULT_CLASS_NAME,
    CONF_WEAVIATE_DEFAULT_MAX_RESULTS,
    CONF_WEAVIATE_DEFAULT_THRESHOLD,
    CONF_WEAVIATE_HOST,
    CONF_WEAVIATE_MAX_RESULTS,
    CONF_WEAVIATE_OPTIONS,
    CONF_WEAVIATE_THRESHOLD,
    DOMAIN,
    LOGGER,
)
from .weaviate import WeaviateClient

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10


def _adjust_schema(schema: dict[str, Any]) -> None:
    """Adjust the schema to be compatible with OpenRouter API."""
    if schema["type"] == "object":
        if "properties" not in schema:
            return

        if "required" not in schema:
            schema["required"] = []

        # Ensure all properties are required
        for prop, prop_info in schema["properties"].items():
            _adjust_schema(prop_info)
            if prop not in schema["required"]:
                prop_info["type"] = [prop_info["type"], "null"]
                schema["required"].append(prop)

    elif schema["type"] == "array":
        if "items" not in schema:
            return

        _adjust_schema(schema["items"])


def _format_structured_output(
    name: str, schema: vol.Schema, llm_api: llm.APIInstance | None
) -> JSONSchema:
    """Format the schema to be compatible with OpenRouter API."""
    result: JSONSchema = {
        "name": name,
        "strict": True,
    }
    result_schema = convert(
        schema,
        custom_serializer=(
            llm_api.custom_serializer if llm_api else llm.selector_serializer
        ),
    )

    _adjust_schema(result_schema)

    result["schema"] = result_schema
    return result


def _format_tool(
    tool: llm.Tool,
    custom_serializer: Callable[[Any], Any] | None,
) -> ChatCompletionFunctionToolParam:
    """Format tool specification."""
    tool_spec = FunctionDefinition(
        name=tool.name,
        parameters=convert(tool.parameters, custom_serializer=custom_serializer),
    )
    tool_spec["description"] = (
        tool.description
        if tool.description is not None and tool.description.strip()
        else "A callable function"
    )
    return ChatCompletionFunctionToolParam(type="function", function=tool_spec)


def b64_file(file_path):
    """Retrieve the base64 encoded file contents."""
    return base64.b64encode(file_path.read_bytes()).decode("utf-8")


async def _convert_content_to_chat_message(
    content: conversation.Content,
) -> ChatCompletionMessageParam | None:
    """Convert any native chat message for this agent to the native format."""
    if isinstance(content, conversation.ToolResultContent):
        return ChatCompletionToolMessageParam(
            role="tool",
            tool_call_id=content.tool_call_id,
            content=json.dumps(content.tool_result),
        )

    role: Literal["user", "assistant", "system"] = content.role
    if role == "system" and content.content:
        return ChatCompletionSystemMessageParam(role="system", content=content.content)

    if role == "user" and content.content:
        messages = []

        if content.attachments:
            loop = asyncio.get_running_loop()
            for attachment in content.attachments or ():
                if not attachment.mime_type.startswith("image/"):
                    raise HomeAssistantError(
                        translation_domain=DOMAIN,
                        translation_key="unsupported_attachment_type",
                    )
                base64_file = await loop.run_in_executor(
                    None, b64_file, attachment.path
                )
                messages.append(
                    ChatCompletionContentPartImageParam(
                        type="image_url",
                        image_url={
                            "url": f"data:{attachment.mime_type};base64,{base64_file}",
                            "detail": "auto",
                        },
                    )
                )

        messages.append(
            ChatCompletionContentPartTextParam(type="text", text=content.content)
        )
        return ChatCompletionUserMessageParam(
            role="user",
            content=messages,
        )

    if role == "assistant":
        param = ChatCompletionAssistantMessageParam(
            role="assistant",
            content=content.content,
        )
        if isinstance(content, conversation.AssistantContent) and content.tool_calls:
            param["tool_calls"] = [
                ChatCompletionMessageFunctionToolCallParam(
                    type="function",
                    id=tool_call.id,
                    function=Function(
                        arguments=json.dumps(tool_call.tool_args),
                        name=tool_call.tool_name,
                    ),
                )
                for tool_call in content.tool_calls
            ]
        return param
    LOGGER.warning("Could not convert message to Completions API: %s", content)
    return None


async def _transform_stream(
    stream: AsyncStream[ChatCompletionChunk],
    strip_emojis: bool,
) -> AsyncGenerator[conversation.AssistantContentDeltaDict, None]:
    """Transform a streaming OpenAI response to ChatLog format."""
    new_msg = True
    pending_think = ""
    in_think = False
    seen_visible = False
    loop = asyncio.get_running_loop()
    pending_tool_calls: list[dict] = []

    async for event in stream:
        chunk: conversation.AssistantContentDeltaDict = {}

        if not event.choices:
            continue

        choice = event.choices[0]
        delta = choice.delta

        if new_msg:
            chunk["role"] = delta.role
            new_msg = False

        if choice.finish_reason and pending_tool_calls:
            chunk["tool_calls"] = [
                llm.ToolInput(
                    tool_name=tool_call["name"],
                    tool_args=json.loads(tool_call["args"])
                    if tool_call["args"]
                    else {},
                )
                for tool_call in pending_tool_calls
            ]
            pending_tool_calls = []

        if (tool_calls := delta.tool_calls) is not None and tool_calls:
            tool_call = tool_calls[0]
            if len(pending_tool_calls) < tool_call.index + 1:
                pending_tool_calls.append(
                    {
                        "name": tool_call.function.name,
                        "args": tool_call.function.arguments or "",
                    }
                )
            else:
                pending_tool_calls[tool_call.index]["args"] += (
                    tool_call.function.arguments
                )

        if (content := delta.content) is not None:
            if strip_emojis:
                content = await loop.run_in_executor(None, demoji.replace, content, "")

            if content == "<think>":
                in_think = True
                pending_think = ""

            if in_think:
                if content == "</think>":
                    in_think = False
                    if pending_think.strip():
                        LOGGER.debug(f"LLM Thought: {pending_think}")
                    pending_think = ""
                elif content != "<think>":
                    pending_think = pending_think + content
            elif content.strip():
                seen_visible = True

            if seen_visible:
                chunk["content"] = content

        if seen_visible or chunk.get("tool_calls") or chunk.get("role"):
            yield chunk


class LocalAiEntity(Entity):
    """Base entity for Open Router."""

    _attr_has_entity_name = True

    def __init__(self, entry: LocalAiConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self.model = subentry.data[CONF_MODEL]
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        structure_name: str | None = None,
        structure: vol.Schema | None = None,
        user_input: conversation.ConversationInput | None = None,
    ) -> None:
        """Generate an answer for the chat log."""
        options = self.subentry.data
        strip_emojis = options.get(CONF_STRIP_EMOJIS)
        max_message_history = options.get(CONF_MAX_MESSAGE_HISTORY, 0)
        temperature = options.get(CONF_TEMPERATURE, 0.6)
        parallel_tool_calls = options.get(CONF_PARALLEL_TOOL_CALLS, True)

        model_args = {
            "model": self.model,
            "user": chat_log.conversation_id,
            "temperature": temperature,
            "parallel_tool_calls": parallel_tool_calls,
        }

        tools: list[ChatCompletionFunctionToolParam] | None = None
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        if tools:
            model_args["tools"] = tools

        messages = self._trim_history(
            [
                m
                for content in chat_log.content
                if (m := await _convert_content_to_chat_message(content))
            ],
            max_message_history,
        )

        # Retrieval Augmented Generation: Query Weaviate vector DB
        try:
            weaviate_opts = options.get(CONF_WEAVIATE_OPTIONS, {})
            weaviate_server_opts = self.entry.data.get(CONF_WEAVIATE_OPTIONS, {})
            weaviate_host = weaviate_server_opts.get(CONF_WEAVIATE_HOST)
            weaviate_class = weaviate_opts.get(
                CONF_WEAVIATE_CLASS_NAME, CONF_WEAVIATE_DEFAULT_CLASS_NAME
            )

            if weaviate_host:
                client = WeaviateClient(
                    hass=self.hass,
                    host=weaviate_host,
                    api_key=weaviate_server_opts.get(CONF_WEAVIATE_API_KEY),
                )
                results = await client.near_text(
                    class_name=weaviate_class,
                    query=user_input.text,
                    threshold=weaviate_opts.get(
                        CONF_WEAVIATE_THRESHOLD, CONF_WEAVIATE_DEFAULT_THRESHOLD
                    ),
                    limit=int(
                        weaviate_opts.get(
                            CONF_WEAVIATE_MAX_RESULTS, CONF_WEAVIATE_DEFAULT_MAX_RESULTS
                        )
                    ),
                )

                LOGGER.debug(f"Weaviate results: {results}")

                result_content = [
                    result.get("content").strip()
                    for result in results
                    if result.get("content", "").strip()
                ]
                if result_content:
                    messages.append(
                        ChatCompletionToolMessageParam(
                            role="tool",
                            content=f"# Retrieval Augmented Generation\nYou may use the following information to answer the user question, if appropriate. Ignore this if it does not relate to or answer the users query.\n\n{'\n'.join(result_content)}",
                            tool_call_id="rag_result",
                        )
                    )

        except Exception as err:
            LOGGER.warning(
                "An unexpected exception occurred while processing RAG: %s", err
            )

        model_args["messages"] = messages

        if structure:
            if TYPE_CHECKING:
                assert structure_name is not None
            model_args["response_format"] = ResponseFormatJSONSchema(
                type="json_schema",
                json_schema=_format_structured_output(
                    structure_name, structure, chat_log.llm_api
                ),
            )

        client = self.entry.runtime_data

        for _iteration in range(MAX_TOOL_ITERATIONS):
            try:
                result_stream = await client.chat.completions.create(
                    **model_args, stream=True
                )
            except openai.OpenAIError as err:
                LOGGER.error("Error requesting response from API: %s", err)
                raise HomeAssistantError("Error talking to API") from err

            try:
                model_args["messages"].extend(
                    [
                        msg
                        async for content in chat_log.async_add_delta_content_stream(
                            self.entity_id,
                            _transform_stream(
                                stream=result_stream, strip_emojis=strip_emojis
                            ),
                        )
                        if (msg := await _convert_content_to_chat_message(content))
                    ]
                )
            except Exception as err:
                LOGGER.error("Error handling API response: %s", err)

            if not chat_log.unresponded_tool_results:
                break

    @staticmethod
    def _trim_history(messages: list, max_messages: int) -> list:
        """
        Trims excess messages from a single history.

        This sets the max history to allow a configurable size history may take
        up in the context window.

        Logic borrowed from the Ollama integration with thanks
        """
        if max_messages < 1:
            # Keep all messages
            return messages

        # Ignore the in progress user message
        num_previous_rounds = sum(m["role"] == "assistant" for m in messages) - 1
        if num_previous_rounds >= max_messages:
            # Trim history but keep system prompt (first message).
            # Every other message should be an assistant message, so keep 2x
            # message objects. Also keep the last in progress user message
            num_keep = 2 * max_messages + 1
            drop_index = len(messages) - num_keep
            messages = [
                messages[0],
                *messages[int(drop_index) :],
            ]

        return messages
