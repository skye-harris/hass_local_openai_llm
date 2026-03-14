"""AI Task integration for Local OpenAI LLM."""

from __future__ import annotations

import base64
import io
import logging
from json import JSONDecodeError

import openai
from homeassistant.components import ai_task, conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util.json import json_loads
from PIL import Image

from . import LocalAiConfigEntry
from .const import (
    CONF_AI_TASK_SUPPORTED_ATTRIBUTE_OPTIONS,
    CONF_AI_TASK_SUPPORTED_ATTRIBUTES,
)
from .entity import LocalAiEntity

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: LocalAiConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up AI Task entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "ai_task_data":
            continue

        async_add_entities(
            [LocalAITaskEntity(config_entry, subentry)],
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
        self, config_entry: LocalAiConfigEntry, subentry: ConfigSubentry
    ) -> None:
        ai_task.AITaskEntity.__init__(self)
        LocalAiEntity.__init__(self, config_entry, subentry)

        supported_attributes = self.subentry.data.get(
            CONF_AI_TASK_SUPPORTED_ATTRIBUTES, ["generate_data"]
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
        await self._async_handle_chat_log(chat_log, task.name, task.structure)

        if not isinstance(chat_log.content[-1], conversation.AssistantContent):
            raise HomeAssistantError(
                "Last content in chat log is not an AssistantContent"
            )

        text = chat_log.content[-1].content or ""

        if not task.structure:
            return ai_task.GenDataTaskResult(
                conversation_id=chat_log.conversation_id,
                data=text,
            )
        try:
            data = json_loads(text)
        except JSONDecodeError as err:
            raise HomeAssistantError("Error with structured response") from err

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
        except Exception as err:
            _LOGGER.error("Error decoding base64 image response: %s", err)
            raise HomeAssistantError(f"Error decoding image response: {err}") from err

        _LOGGER.debug(
            "Generated image details: mime_type=%s, width=%s, height=%s",
            "image/png",
            width,
            height,
        )

        return ai_task.GenImageTaskResult(
            image_data=img_data,
            conversation_id=chat_log.conversation_id,
            mime_type="image/png",
            width=width,
            height=height,
            model=self.model,
        )

    async def _async_handle_image_response(
        self,
        ai_task: ai_task.GenImageTask,
    ):
        """Generate an image response using the Responses API."""
        client = self.entry.runtime_data

        _LOGGER.debug("Sending image generation request to API")
        try:
            response = await client.images.generate(
                prompt=ai_task.instructions, model=self.model, output_format="png"
            )
        except openai.OpenAIError as err:
            _LOGGER.error("Error requesting image response from API: %s", err)
            raise HomeAssistantError(f"Error talking to API: {err}") from err

        if len(response.data) == 0:
            _LOGGER.debug("No image received from API: %s")
            raise HomeAssistantError("No image was returned by the API")

        return response.data[0]
