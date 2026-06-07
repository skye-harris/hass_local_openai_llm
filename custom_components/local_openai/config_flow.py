"""Config flow for Local OpenAI LLM integration."""

from __future__ import annotations

import re
from typing import Any

import voluptuous as vol
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_MODEL, CONF_PROMPT
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import SectionConfig, section
from homeassistant.helpers import llm
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    ObjectSelector,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
)
from openai import AsyncOpenAI, OpenAIError

from custom_components.local_openai.entities.deepseek import (
    get_conversation_config_schema as _deepseek_conversation_schema,
)
from custom_components.local_openai.entities.llama_cpp import (
    get_ai_task_config_schema as _llama_cpp_ai_task_schema,
)
from custom_components.local_openai.entities.llama_cpp import (
    get_conversation_config_schema as _llama_cpp_conversation_schema,
)
from custom_components.local_openai.entities.llama_cpp import (
    get_model_alias as _llama_cpp_model_alias,
)

from .const import (
    CONF_AI_TASK_SUPPORTED_ATTRIBUTES,
    CONF_AI_TASK_TOOLS_SECTION,
    CONF_BASE_URL,
    CONF_CHAT_TEMPLATE_KWARGS,
    CONF_CHAT_TEMPLATE_OPTS,
    CONF_CONTENT_INJECTION_METHOD,
    CONF_CONTENT_INJECTION_METHODS,
    CONF_DEEPSEEK_CONFIG,
    CONF_GENERIC_CONFIG,
    CONF_LLAMACPP_CONFIG,
    CONF_MAX_MESSAGE_HISTORY,
    CONF_PARALLEL_TOOL_CALLS,
    CONF_PASS_SESSION_ID,
    CONF_SERVER_NAME,
    CONF_SERVER_OPTIONS,
    CONF_SERVER_TYPE,
    CONF_STRIP_EMOJIS,
    CONF_TEMPERATURE,
    CONF_VLLM_CONFIG,
    CONF_WEAVIATE_API_KEY,
    CONF_WEAVIATE_CLASS_NAME,
    CONF_WEAVIATE_DEFAULT_CLASS_NAME,
    CONF_WEAVIATE_DEFAULT_HYBRID_SEARCH_ALPHA,
    CONF_WEAVIATE_DEFAULT_MAX_RESULTS,
    CONF_WEAVIATE_DEFAULT_THRESHOLD,
    CONF_WEAVIATE_HOST,
    CONF_WEAVIATE_HYBRID_SEARCH_ALPHA,
    CONF_WEAVIATE_MAX_RESULTS,
    CONF_WEAVIATE_MAX_RESULTS_MAX,
    CONF_WEAVIATE_OPTIONS,
    CONF_WEAVIATE_THRESHOLD,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CONVERSATION_OPTIONS,
    SERVER_TYPE_DEEPSEEK,
    SERVER_TYPE_GENERIC,
    SERVER_TYPE_LLAMACPP,
    SERVER_TYPE_OPTIONS,
    SERVER_TYPE_VLLM,
)
from .weaviate import WeaviateClient, WeaviateError


async def prepare_weaviate_class(
    hass: HomeAssistant,
    weaviate_opts: dict[str, Any],
) -> None:
    """Prepare our object class."""
    host = weaviate_opts.get(CONF_WEAVIATE_HOST)
    if not host:
        # Just pass if we dont have a weaviate host defined
        return

    weaviate = WeaviateClient(
        hass=hass,
        host=host,
        api_key=weaviate_opts.get(CONF_WEAVIATE_API_KEY),
    )

    class_name = weaviate_opts.get(
        CONF_WEAVIATE_CLASS_NAME,
        CONF_WEAVIATE_DEFAULT_CLASS_NAME,
    )

    # if the class already exists, we're good
    if await weaviate.does_class_exist(class_name):
        return

    await weaviate.create_class(class_name)
    LOGGER.debug("Weaviate connectivity confirmed and class is prepared")


def options_to_selections_dict(opts: dict) -> list[SelectOptionDict]:
    """Convert a dict to a list of select options."""
    return [SelectOptionDict(value=key, label=opts[key]) for key in opts]


def _get_conversation_config_schema(server_type: str) -> dict:
    """Get the server-specific config fields for Conversation Agent entities."""
    provider = {
        SERVER_TYPE_DEEPSEEK: _deepseek_conversation_schema,
        SERVER_TYPE_LLAMACPP: _llama_cpp_conversation_schema,
    }.get(server_type)
    return provider() if provider else {}


def _get_ai_task_config_schema(server_type: str) -> dict:
    """Get the server-specific config fields for AI Task entities."""
    provider = {
        SERVER_TYPE_DEEPSEEK: _deepseek_conversation_schema,
        SERVER_TYPE_LLAMACPP: _llama_cpp_ai_task_schema,
    }.get(server_type)
    return provider() if provider else {}


def _get_server_type_config_key(server_type: str) -> str:
    """Return the config key for the given server type."""
    return {
        SERVER_TYPE_GENERIC: CONF_GENERIC_CONFIG,
        SERVER_TYPE_LLAMACPP: CONF_LLAMACPP_CONFIG,
        SERVER_TYPE_VLLM: CONF_VLLM_CONFIG,
        SERVER_TYPE_DEEPSEEK: CONF_DEEPSEEK_CONFIG,
    }.get(server_type, CONF_GENERIC_CONFIG)


def _resolve_model_name(server_type: str, model: object) -> str:
    """
    Resolve a server-specific display name for a model picker entry.

    Prefer a server-provided alias when one is available; otherwise fall back to the
    raw model ``id``, stripping any file path and ``.gguf`` extension it may contain.
    """
    resolver = {
        SERVER_TYPE_LLAMACPP: _llama_cpp_model_alias,
    }.get(server_type)
    alias = resolver(model) if resolver else None
    return alias or LocalAiSubentryFlowHandler.strip_model_pathing(model.id)


class LocalAiConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Local OpenAI LLM."""

    VERSION = 2

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls,
        _config_entry: ConfigEntry,
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this handler."""
        return {
            "conversation": ConversationFlowHandler,
            "ai_task_data": AITaskDataFlowHandler,
        }

    @staticmethod
    def get_schema() -> vol.Schema:
        """Get the schema for the config flow form."""
        return vol.Schema(
            {
                vol.Required(
                    CONF_SERVER_NAME,
                    default="Local LLM Server",
                ): str,
                vol.Required(CONF_BASE_URL, default=""): str,
                vol.Optional(CONF_API_KEY, default=""): str,
                vol.Required(
                    CONF_SERVER_TYPE,
                    default=SERVER_TYPE_GENERIC,
                ): SelectSelector(
                    SelectSelectorConfig(
                        mode=SelectSelectorMode.DROPDOWN,
                        options=options_to_selections_dict(SERVER_TYPE_OPTIONS),
                    ),
                ),
                vol.Optional(CONF_SERVER_OPTIONS): section(
                    schema=vol.Schema(
                        schema={
                            vol.Optional(
                                CONF_PASS_SESSION_ID,
                                default=False,
                            ): bool,
                        },
                    ),
                    options={"collapsed": True},
                ),
                vol.Optional(CONF_WEAVIATE_OPTIONS): section(
                    schema=vol.Schema(
                        schema={
                            vol.Optional(
                                CONF_WEAVIATE_HOST,
                                default="",
                            ): str,
                            vol.Optional(
                                CONF_WEAVIATE_API_KEY,
                                default="",
                            ): str,
                        },
                    ),
                    options={"collapsed": True},
                ),
            },
        )

    async def async_step_user(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors = {}
        if user_input is not None:
            self._async_abort_entries_match(user_input)
            LOGGER.debug(
                f"Initialising OpenAI client with base_url: {user_input[CONF_BASE_URL]}",
            )

            try:
                client = AsyncOpenAI(
                    base_url=user_input.get(CONF_BASE_URL),
                    api_key=user_input.get(CONF_API_KEY, ""),
                    http_client=get_async_client(self.hass),
                )

                LOGGER.debug("Retrieving model list to ensure server is accessible")
                await client.models.list()

                # Test connectivity with Weaviate
                await prepare_weaviate_class(
                    hass=self.hass,
                    weaviate_opts=user_input.get(CONF_WEAVIATE_OPTIONS, {}),
                )
            except WeaviateError as err:
                LOGGER.exception(f"Unexpected exception: {err}")
                errors["base"] = "cannot_connect_weaviate"
            except OpenAIError as err:
                LOGGER.exception(f"OpenAI Error: {err}")
                errors["base"] = "cannot_connect"
            except Exception as err:
                LOGGER.exception(f"Unexpected exception: {err}")
                errors["base"] = "unknown"
            else:
                LOGGER.debug("Server connection verified")

                return self.async_create_entry(
                    title=f"{user_input.get(CONF_SERVER_NAME, 'Local LLM Server')}",
                    data=user_input,
                )

        return self.async_show_form(
            step_id="user",
            data_schema=self.get_schema(),
            errors=errors,
        )

    async def async_step_reconfigure(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> SubentryFlowResult:
        """User flow to create a sensor subentry."""
        errors = {}
        if user_input is not None:
            self._async_abort_entries_match(user_input)
            LOGGER.debug(
                f"Initialising OpenAI client with base_url: {user_input[CONF_BASE_URL]}",
            )

            try:
                client = AsyncOpenAI(
                    base_url=user_input.get(CONF_BASE_URL),
                    api_key=user_input.get(CONF_API_KEY, ""),
                    http_client=get_async_client(self.hass),
                )

                LOGGER.debug("Retrieving model list to ensure server is accessible")
                await client.models.list()

                # Test connectivity with Weaviate
                await prepare_weaviate_class(
                    hass=self.hass,
                    weaviate_opts=user_input.get(CONF_WEAVIATE_OPTIONS, {}),
                )
            except WeaviateError as err:
                LOGGER.exception(f"Unexpected exception: {err}")
                errors["base"] = "cannot_connect_weaviate"
            except OpenAIError as err:
                LOGGER.exception(f"OpenAI Error: {err}")
                errors["base"] = "cannot_connect"
            except Exception as err:
                LOGGER.exception(f"Unexpected exception: {err}")
                errors["base"] = "unknown"
            else:
                LOGGER.debug("Server connection verified")

                return self.async_update_reload_and_abort(
                    entry=self._get_reconfigure_entry(),
                    title=f"{user_input.get(CONF_SERVER_NAME, 'Local LLM Server')}",
                    data=user_input,
                )

        options = self._get_reconfigure_entry().data.copy()
        schema = self.add_suggested_values_to_schema(self.get_schema(), options)

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=schema,
            errors=errors,
        )


class LocalAiSubentryFlowHandler(ConfigSubentryFlow):
    """Handle subentry flow for Local OpenAI LLM."""

    def get_llm_apis(self) -> list[SelectOptionDict]:
        """Get available LLM APIs as select options."""
        return [
            SelectOptionDict(
                label=api.name,
                value=api.id,
            )
            for api in llm.async_get_apis(self.hass)
        ]

    @staticmethod
    def strip_model_pathing(model_name: str) -> str:
        """llama.cpp at the very least will keep the full model file path supplied from the CLI so lets look to strip that and any .gguf extension."""
        matches = re.search(r"([^\/]*)\.gguf$", model_name.strip())
        return matches[1] if matches else model_name


class ConversationFlowHandler(LocalAiSubentryFlowHandler):
    """Handle subentry flow."""

    async def get_schema(self) -> vol.Schema:
        """Get the schema for the conversation subentry form."""
        llm_apis = self.get_llm_apis()
        entry = self._get_entry()
        client = entry.runtime_data
        server_type = entry.data.get(CONF_SERVER_TYPE, SERVER_TYPE_GENERIC)

        try:
            response = await client.models.list()
            downloaded_models: list[SelectOptionDict] = [
                SelectOptionDict(
                    label=name,
                    value=name,
                )
                for model in response.data
                if (name := _resolve_model_name(server_type, model))
            ]
        except OpenAIError as err:
            LOGGER.exception(f"OpenAI Error retrieving models list: {err}")
            downloaded_models = []
        except Exception as err:
            LOGGER.exception(f"Unexpected exception retrieving models list: {err}")
            downloaded_models = []

        schema = {
            vol.Required(
                CONF_MODEL,
            ): SelectSelector(
                SelectSelectorConfig(options=downloaded_models, custom_value=True),
            ),
            vol.Optional(
                CONF_PROMPT,
                default=RECOMMENDED_CONVERSATION_OPTIONS[CONF_PROMPT],
            ): TemplateSelector(),
            vol.Optional(
                CONF_LLM_HASS_API,
                default=RECOMMENDED_CONVERSATION_OPTIONS[CONF_LLM_HASS_API],
            ): SelectSelector(SelectSelectorConfig(options=llm_apis, multiple=True)),
            vol.Required(
                CONF_PARALLEL_TOOL_CALLS,
                default=True,
            ): bool,
            vol.Required(
                CONF_STRIP_EMOJIS,
                default=False,
            ): bool,
            vol.Required(
                CONF_TEMPERATURE,
                default=0.6,
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0,
                    max=1,
                    step=0.01,
                    mode=NumberSelectorMode.BOX,
                ),
            ),
            vol.Optional(
                CONF_MAX_MESSAGE_HISTORY,
                default=0,
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0,
                    max=50,
                    step=1,
                    mode=NumberSelectorMode.BOX,
                ),
            ),
            vol.Optional(
                CONF_CONTENT_INJECTION_METHOD,
            ): SelectSelector(
                SelectSelectorConfig(
                    mode=SelectSelectorMode.DROPDOWN,
                    options=CONF_CONTENT_INJECTION_METHODS,
                ),
            ),
            vol.Required(CONF_CHAT_TEMPLATE_OPTS): section(
                options=SectionConfig(collapsed=True),
                schema=vol.Schema(
                    schema={
                        vol.Required(
                            CONF_CHAT_TEMPLATE_KWARGS,
                            default=[],
                        ): ObjectSelector(
                            config={
                                "multiple": True,
                                "fields": {
                                    "Key": {
                                        "selector": {"text": None},
                                        "required": True,
                                    },
                                    "Value": {
                                        "selector": {"template": None},
                                        "required": True,
                                    },
                                },
                            },
                        ),
                    },
                ),
            ),
        }

        server_type_schema_fields = _get_conversation_config_schema(server_type)
        if server_type_schema_fields:
            schema = {
                **schema,
                vol.Required(_get_server_type_config_key(server_type)): section(
                    options=SectionConfig(collapsed=True),
                    schema=vol.Schema(schema=server_type_schema_fields),
                ),
            }

        if entry.data.get(CONF_WEAVIATE_OPTIONS, {}).get(CONF_WEAVIATE_HOST):
            schema = {
                **schema,
                vol.Optional(CONF_WEAVIATE_OPTIONS): section(
                    schema=vol.Schema(
                        schema={
                            vol.Optional(
                                CONF_WEAVIATE_CLASS_NAME,
                                default=CONF_WEAVIATE_DEFAULT_CLASS_NAME,
                            ): str,
                            vol.Optional(
                                CONF_WEAVIATE_MAX_RESULTS,
                                default=CONF_WEAVIATE_DEFAULT_MAX_RESULTS,
                            ): NumberSelector(
                                NumberSelectorConfig(
                                    min=1,
                                    max=CONF_WEAVIATE_MAX_RESULTS_MAX,
                                    step=1,
                                    mode=NumberSelectorMode.SLIDER,
                                ),
                            ),
                            vol.Optional(
                                CONF_WEAVIATE_THRESHOLD,
                                default=CONF_WEAVIATE_DEFAULT_THRESHOLD,
                            ): NumberSelector(
                                NumberSelectorConfig(
                                    min=0,
                                    max=1,
                                    step=0.01,
                                    mode=NumberSelectorMode.SLIDER,
                                ),
                            ),
                            vol.Optional(
                                CONF_WEAVIATE_HYBRID_SEARCH_ALPHA,
                                default=CONF_WEAVIATE_DEFAULT_HYBRID_SEARCH_ALPHA,
                            ): NumberSelector(
                                NumberSelectorConfig(
                                    min=0,
                                    max=1,
                                    step=0.01,
                                    mode=NumberSelectorMode.SLIDER,
                                ),
                            ),
                        },
                    ),
                    options=SectionConfig(collapsed=True),
                ),
            }

        return vol.Schema(schema)

    async def async_step_user(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> SubentryFlowResult:
        """User flow to create a sensor subentry."""
        errors = {}

        if user_input is not None:
            if not user_input.get(CONF_LLM_HASS_API):
                user_input.pop(CONF_LLM_HASS_API, None)
            model_name = self.strip_model_pathing(user_input.get(CONF_MODEL, "Local"))

            try:
                weaviate_opts = {
                    **self._get_entry().data.get(CONF_WEAVIATE_OPTIONS, {}),
                    **user_input.get(CONF_WEAVIATE_OPTIONS, {}),
                }
                await prepare_weaviate_class(
                    hass=self.hass,
                    weaviate_opts=weaviate_opts,
                )
            except WeaviateError as err:
                LOGGER.exception(f"Unexpected exception: {err}")
                errors["base"] = "cannot_connect_weaviate"
            else:
                server_name = self._get_entry().title
                return self.async_create_entry(
                    title=f"{server_name}: {model_name} AI Agent",
                    data=user_input,
                )

        return self.async_show_form(
            step_id="user",
            data_schema=await self.get_schema(),
            errors=errors,
        )

    async def async_step_reconfigure(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> SubentryFlowResult:
        """User flow to create a sensor subentry."""
        errors = {}

        if user_input is not None:
            if not user_input.get(CONF_LLM_HASS_API):
                user_input.pop(CONF_LLM_HASS_API, None)

            try:
                weaviate_opts = {
                    **self._get_entry().data.get(CONF_WEAVIATE_OPTIONS, {}),
                    **user_input.get(CONF_WEAVIATE_OPTIONS, {}),
                }
                await prepare_weaviate_class(
                    hass=self.hass,
                    weaviate_opts=weaviate_opts,
                )
            except WeaviateError as err:
                LOGGER.exception(f"Unexpected exception: {err}")
                errors["base"] = "cannot_connect_weaviate"
            else:
                return self.async_update_and_abort(
                    self._get_entry(),
                    self._get_reconfigure_subentry(),
                    data=user_input,
                )

        options = self._get_reconfigure_subentry().data.copy()

        # Filter out any tool providers that no longer exist
        llm_apis = [api.id for api in llm.async_get_apis(self.hass)]

        options[CONF_LLM_HASS_API] = [
            api for api in options.get(CONF_LLM_HASS_API, []) if api in llm_apis
        ]

        schema = self.add_suggested_values_to_schema(await self.get_schema(), options)

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=schema,
            errors=errors,
        )


class AITaskDataFlowHandler(LocalAiSubentryFlowHandler):
    """Handle subentry flow."""

    async def get_schema(self) -> vol.Schema:
        """Get the schema for the AI task data subentry form."""
        entry = self._get_entry()
        server_type = entry.data.get(CONF_SERVER_TYPE, SERVER_TYPE_GENERIC)
        try:
            client = entry.runtime_data
            response = await client.models.list()
            downloaded_models: list[SelectOptionDict] = [
                SelectOptionDict(
                    label=name,
                    value=name,
                )
                for model in response.data
                if (name := _resolve_model_name(server_type, model))
            ]
        except OpenAIError as err:
            LOGGER.exception(f"OpenAI Error retrieving models list: {err}")
            downloaded_models = []
        except Exception as err:
            LOGGER.exception(f"Unexpected exception retrieving models list: {err}")
            downloaded_models = []

        llm_apis = self.get_llm_apis()

        schema = {
            vol.Required(
                CONF_MODEL,
            ): SelectSelector(
                SelectSelectorConfig(options=downloaded_models, custom_value=True),
            ),
            vol.Required(
                CONF_AI_TASK_SUPPORTED_ATTRIBUTES,
            ): SelectSelector(
                SelectSelectorConfig(
                    options=[
                        {"value": "generate_data", "label": "Generate Data"},
                        {"value": "generate_image", "label": "Generate Image"},
                    ],
                    multiple=True,
                    mode=SelectSelectorMode.LIST,
                ),
            ),
            vol.Required(CONF_AI_TASK_TOOLS_SECTION): section(
                options=SectionConfig(collapsed=True),
                schema=vol.Schema(
                    schema={
                        vol.Optional(
                            CONF_LLM_HASS_API,
                            default=[],
                        ): SelectSelector(
                            SelectSelectorConfig(options=llm_apis, multiple=True),
                        ),
                        vol.Required(
                            CONF_PARALLEL_TOOL_CALLS,
                            default=True,
                        ): bool,
                    },
                ),
            ),
            vol.Required(CONF_CHAT_TEMPLATE_OPTS): section(
                options=SectionConfig(collapsed=True),
                schema=vol.Schema(
                    schema={
                        vol.Required(
                            CONF_CHAT_TEMPLATE_KWARGS,
                            default=[],
                        ): ObjectSelector(
                            config={
                                "multiple": True,
                                "fields": {
                                    "Key": {
                                        "selector": {"text": None},
                                        "required": True,
                                    },
                                    "Value": {
                                        "selector": {"template": None},
                                        "required": True,
                                    },
                                },
                            },
                        ),
                    },
                ),
            ),
        }

        server_type_schema_fields = _get_ai_task_config_schema(server_type)
        if server_type_schema_fields:
            schema = {
                **schema,
                vol.Required(_get_server_type_config_key(server_type)): section(
                    options=SectionConfig(collapsed=True),
                    schema=vol.Schema(schema=server_type_schema_fields),
                ),
            }

        return vol.Schema(schema)

    async def async_step_user(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> SubentryFlowResult:
        """User flow to create a sensor subentry."""
        if user_input is not None:
            model_name = self.strip_model_pathing(user_input.get(CONF_MODEL, "Local"))
            server_name = self._get_entry().title
            return self.async_create_entry(
                title=f"{server_name}: {model_name} AI Task",
                data=user_input,
            )

        schema = await self.get_schema()
        return self.async_show_form(step_id="user", data_schema=schema)

    async def async_step_reconfigure(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> SubentryFlowResult:
        """User flow to create a sensor subentry."""
        errors = {}
        if user_input is not None:
            return self.async_update_and_abort(
                entry=self._get_entry(),
                subentry=self._get_reconfigure_subentry(),
                data=user_input,
            )

        options = self._get_reconfigure_subentry().data.copy()
        schema = self.add_suggested_values_to_schema(await self.get_schema(), options)

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=schema,
            errors=errors,
        )
