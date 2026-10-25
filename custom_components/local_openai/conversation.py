"""Conversation support for Local OpenAI LLM."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from homeassistant.components import conversation
from homeassistant.const import CONF_LLM_HASS_API, CONF_PROMPT, MATCH_ALL
from homeassistant.helpers import llm

from .const import (
    CONF_ALWAYS_CONTINUE_CONVERSATION,
    CONF_ALWAYS_CONTINUE_CONVERSATION_DEFAULT,
    CONF_PARALLEL_TOOL_CALLS,
    CONF_SERVER_TYPE,
    DOMAIN,
    SERVER_TYPE_DEEPSEEK,
    SERVER_TYPE_GENERIC,
    SERVER_TYPE_GOOGLE_GEMINI,
    SERVER_TYPE_LLAMACPP,
    SERVER_TYPE_LOCALAI,
    SERVER_TYPE_VLLM,
)
from .entity import LocalAiEntity
from .tools.api import InternalToolApi
from .tools.stop_talking import StopTalkingTool

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigSubentry
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

    from . import LocalAiConfigEntry


def _get_conversation_entity(
    server_type: str,
) -> type[LocalAiConversationEntity]:
    if getattr(_get_conversation_entity, "entity_map", None) is None:
        from .entities.deepseek import DeepSeekConversationEntity  # noqa: PLC0415
        from .entities.google_gemini import (  # noqa: PLC0415
            GoogleGeminiConversationEntity,
        )
        from .entities.llama_cpp import LlamaCppConversationEntity  # noqa: PLC0415
        from .entities.localai import LocalAIServerConversationEntity  # noqa: PLC0415
        from .entities.vllm import VllmConversationEntity  # noqa: PLC0415

        _get_conversation_entity.entity_map = {
            SERVER_TYPE_DEEPSEEK: DeepSeekConversationEntity,
            SERVER_TYPE_GOOGLE_GEMINI: GoogleGeminiConversationEntity,
            SERVER_TYPE_LLAMACPP: LlamaCppConversationEntity,
            SERVER_TYPE_LOCALAI: LocalAIServerConversationEntity,
            SERVER_TYPE_VLLM: VllmConversationEntity,
        }

    return _get_conversation_entity.entity_map.get(
        server_type,
        LocalAiConversationEntity,
    )


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: LocalAiConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    entity_cls = _get_conversation_entity(
        config_entry.data.get(CONF_SERVER_TYPE, SERVER_TYPE_GENERIC),
    )
    for subentry_id, subentry in config_entry.subentries.items():
        if subentry.subentry_type != "conversation":
            continue
        async_add_entities(
            [entity_cls(config_entry, subentry)],
            config_subentry_id=subentry_id,
        )


class LocalAiConversationEntity(LocalAiEntity, conversation.ConversationEntity):
    """Local OpenAI LLM conversation agent."""

    _attr_name = None
    _attr_supports_streaming = True

    def __init__(self, entry: LocalAiConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the agent."""
        super().__init__(entry, subentry)
        if self.subentry.data.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Process the user input and call the API."""
        options = self.subentry.data
        system_prompt = options.get(CONF_PROMPT)
        parallel_tool_calls = options.get(CONF_PARALLEL_TOOL_CALLS, True)
        always_continue_conversation = options.get(
            CONF_ALWAYS_CONTINUE_CONVERSATION, CONF_ALWAYS_CONTINUE_CONVERSATION_DEFAULT
        )

        hass_apis = [api.id for api in llm.async_get_apis(self.hass)]

        # Filter out any tool providers that no longer exist
        llm_apis = options.get(CONF_LLM_HASS_API, [])
        llm_apis = [api for api in llm_apis if api in hass_apis]

        if always_continue_conversation and not llm_apis:
            # Inject our internal tooling for `stop_talking` as an API if no other tooling providers exist
            llm_apis = InternalToolApi(
                hass=self.hass,
                name="DOMAIN",
            )

        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                llm_apis,
                system_prompt,
                user_input.extra_system_prompt,
            )

            if always_continue_conversation and not isinstance(
                llm_apis, InternalToolApi
            ):
                # Append our stop_talking tool to the existing tooling if tooling APIs were configured
                chat_log.llm_api.tools.append(StopTalkingTool())
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        halt_conversation = await self._async_handle_chat_log(
            chat_log,
            user_input=user_input,
            parallel_tool_calls=parallel_tool_calls,
        )

        chat_log_result = conversation.async_get_result_from_chat_log(
            user_input, chat_log
        )

        if halt_conversation:
            # If we've received our stop_talking halt signal
            chat_log_result.continue_conversation = False
        elif always_continue_conversation:
            chat_log_result.continue_conversation = True

        return chat_log_result
