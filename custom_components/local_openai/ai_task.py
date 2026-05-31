"""AI Task integration for Local OpenAI LLM."""

from __future__ import annotations

import base64
import io
import logging
import re
from json import JSONDecodeError
from typing import TYPE_CHECKING

import openai
from homeassistant.components import ai_task, conversation
from homeassistant.components.conversation import SystemContent
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers.llm import async_get_api, selector_serializer
from homeassistant.util.json import json_loads
from PIL import Image

from . import DOMAIN, LocalAiConfigEntry
from .const import (
    CONF_AI_TASK_SUPPORTED_ATTRIBUTE_OPTIONS,
    CONF_AI_TASK_SUPPORTED_ATTRIBUTES,
    CONF_AI_TASK_TOOLS_SECTION,
    CONF_PARALLEL_TOOL_CALLS,
    CONF_SERVER_TYPE,
    SERVER_TYPE_DEEPSEEK,
    SERVER_TYPE_GENERIC,
    SERVER_TYPE_LLAMACPP,
)
from .entity import LocalAiEntity

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigSubentry
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

_LOGGER = logging.getLogger(__name__)


def _get_ai_task_entity(
    server_type: str,
) -> type[LocalAITaskEntity]:
    if getattr(_get_ai_task_entity, "entity_map", None) is None:
        from .entities.deepseek import DeepSeekAITaskEntity  # noqa: PLC0415
        from .entities.llama_cpp import LlamaCppAITaskEntity  # noqa: PLC0415

        _get_ai_task_entity.entity_map = {
            SERVER_TYPE_DEEPSEEK: DeepSeekAITaskEntity,
            SERVER_TYPE_LLAMACPP: LlamaCppAITaskEntity,
        }
    return _get_ai_task_entity.entity_map.get(server_type, LocalAITaskEntity)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: LocalAiConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up AI Task entities."""
    entity_cls = _get_ai_task_entity(
        config_entry.data.get(CONF_SERVER_TYPE, SERVER_TYPE_GENERIC),
    )
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "ai_task_data":
            continue

        async_add_entities(
            [entity_cls(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class LocalAITaskEntity(
    ai_task.AITaskEntity,
    LocalAiEntity,
):
    """Local OpenAI LLM AI Task entity."""

    _attr_name = None
    _attr_supported_features = (
        ai_task.AITaskEntityFeature.GENERATE_DATA
        | ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS
        | ai_task.AITaskEntityFeature.GENERATE_IMAGE
    )

    def __init__(
        self,
        config_entry: LocalAiConfigEntry,
        subentry: ConfigSubentry,
    ) -> None:
        """Initialize the AI Task entity."""
        ai_task.AITaskEntity.__init__(self)
        LocalAiEntity.__init__(self, config_entry, subentry)

        supported_attributes = self.subentry.data.get(
            CONF_AI_TASK_SUPPORTED_ATTRIBUTES,
            ["generate_data"],
        )
        attributes = 0
        for attr in supported_attributes:
            attributes |= CONF_AI_TASK_SUPPORTED_ATTRIBUTE_OPTIONS[attr]

        self._attr_supported_features = attributes

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task."""
        options = self.subentry.data

        tools_opts = options.get(CONF_AI_TASK_TOOLS_SECTION, {})
        llm_apis = tools_opts.get(CONF_LLM_HASS_API, [])
        parallel_tool_calls = tools_opts.get(CONF_PARALLEL_TOOL_CALLS, False)

        # Inject LLM APIs into AI Data flow
        if llm_apis:
            llm_context = llm.LLMContext(
                platform=DOMAIN,
                context=None,
                language=None,
                assistant=None,
                device_id=None,
            )
            apis = await async_get_api(self.hass, llm_apis, llm_context)
            apis.custom_serializer = apis.custom_serializer or selector_serializer
            chat_log.llm_api = apis
            api_prompt = f"\n\n{apis.api_prompt.strip()}\n\nCurrent time"
            system_prompt = re.sub(
                "\nCurrent time",
                api_prompt,
                chat_log.content[0].content,
            )
            chat_log.content[0] = SystemContent(content=system_prompt)

        await self._async_handle_chat_log(
            chat_log=chat_log,
            structure_name=task.name,
            structure=task.structure,
            parallel_tool_calls=parallel_tool_calls,
        )

        if not isinstance(chat_log.content[-1], conversation.AssistantContent):
            msg = "Last content in chat log is not an AssistantContent"
            raise HomeAssistantError(msg)

        text = chat_log.content[-1].content or ""

        if not task.structure:
            return ai_task.GenDataTaskResult(
                conversation_id=chat_log.conversation_id,
                data=text,
            )
        try:
            data = json_loads(text)
        except JSONDecodeError as err:
            msg = "Error with structured response"
            raise HomeAssistantError(msg) from err

        return ai_task.GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=data,
        )

    async def _async_generate_image(
        self,
        task: ai_task.GenImageTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenImageTaskResult:
        """Handle a generate image task."""
        image = await self._async_handle_image_response(task)
        base64_string = image.b64_json

        if base64_string.startswith("data:image/"):
            base64_string = base64_string.split(",")[1]

        try:
            img_data = base64.b64decode(base64_string)
            image_buffer = io.BytesIO(img_data)
            img = Image.open(image_buffer)
            width, height = img.size
            mime_type = img.get_format_mimetype()
        except Exception as err:
            _LOGGER.exception("Error decoding base64 image response")
            msg = f"Error decoding image response: {err}"
            raise HomeAssistantError(msg) from err

        _LOGGER.debug(
            "Generated image details: mime_type=%s, width=%s, height=%s",
            mime_type,
            width,
            height,
        )

        return ai_task.GenImageTaskResult(
            image_data=img_data,
            conversation_id=chat_log.conversation_id,
            mime_type=mime_type,
            width=width,
            height=height,
            model=self.model,
        )

    async def _async_handle_image_response(
        self,
        ai_task: ai_task.GenImageTask,
    ) -> openai.types.Image:
        """Generate an image response using the Images API."""
        client = self.entry.runtime_data
        attachments = ai_task.attachments or []

        for attachment in attachments:
            if not attachment.mime_type.startswith("image/"):
                msg = f"Unsupported attachment type for image generation: {attachment.mime_type}"
                raise HomeAssistantError(msg)

        _LOGGER.debug("Sending image generation request to API")
        try:
            if attachments:

                def read_files() -> list[tuple[str, bytes, str]]:
                    return [
                        (a.path.name, a.path.read_bytes(), a.mime_type)
                        for a in attachments
                    ]

                files = await self.hass.async_add_executor_job(read_files)
                image_arg = files[0] if len(files) == 1 else files

                response = await client.images.edit(
                    image=image_arg,
                    prompt=ai_task.instructions,
                    model=self.model,
                    response_format="b64_json",
                )
            else:
                response = await client.images.generate(
                    prompt=ai_task.instructions,
                    model=self.model,
                    response_format="b64_json",
                )
        except openai.OpenAIError as err:
            _LOGGER.exception("Error requesting image response from API")
            msg = f"Error talking to API: {err}"
            raise HomeAssistantError(msg) from err

        if len(response.data) == 0:
            _LOGGER.debug("No image received from API")
            msg = "No image was returned by the API"
            raise HomeAssistantError(msg)

        return response.data[0]
