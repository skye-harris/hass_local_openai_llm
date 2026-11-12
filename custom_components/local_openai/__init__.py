"""The Local OpenAI LLM integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import voluptuous as vol
from homeassistant.components.conversation import DOMAIN as CONVERSATION_DOMAIN
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.exceptions import ConfigEntryError, ConfigEntryNotReady
from homeassistant.helpers import service
from homeassistant.helpers.httpx_client import get_async_client
from openai import AsyncOpenAI, AuthenticationError, OpenAIError

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity import Entity
    from homeassistant.helpers.typing import ConfigType

from .const import (
    CONF_BASE_URL,
    CONF_CHAT_TEMPLATE_KWARGS,
    CONF_CHAT_TEMPLATE_OPTS,
    CONF_CONTENT_INJECTION_METHOD,
    CONF_CONTEXT_INJECTION_OPTS,
    CONF_CONTEXT_INJECTION_TEMPLATE,
    CONF_CUSTOM_HEADERS,
    CONF_DEEPSEEK_CONFIG,
    CONF_DEEPSEEK_REASONING_EFFORT,
    CONF_DEEPSEEK_REASONING_EFFORT_DEFAULT,
    CONF_DEEPSEEK_THINKING_MODE,
    CONF_DEEPSEEK_THINKING_MODE_ALWAYS,
    CONF_LLAMACPP_CONFIG,
    CONF_LLAMACPP_ENABLE_THINKING,
    CONF_LLAMACPP_THINKING_MODE,
    CONF_LLAMACPP_THINKING_MODE_ALWAYS,
    CONF_LLAMACPP_THINKING_MODE_DISABLED,
    CONF_MAX_MESSAGE_HISTORY,
    CONF_SERVER_HEADERS,
    CONF_SERVER_TYPE,
    CONF_VLLM_CONFIG,
    CONF_VLLM_THINKING_MODE,
    CONF_VLLM_THINKING_MODE_ALWAYS,
    CONF_VLLM_THINKING_TOKEN_BUDGET,
    DOMAIN,
    LOGGER,
    PLACEHOLDER_API_KEY,
    SERVER_TYPE_DEEPSEEK,
)

PLATFORMS = [Platform.AI_TASK, Platform.CONVERSATION]

type LocalAiConfigEntry = ConfigEntry[AsyncOpenAI]


MIGRATION_FROM_VERSION = "Migrating configuration from version %s.%s"
MIGRATION_SUCCESS_VERSION = "Migration to configuration version %s successful"


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate config entry to new version."""
    LOGGER.debug(MIGRATION_FROM_VERSION, entry.version, entry.minor_version or 1)

    if entry.version == 1:
        await _migrate_v1_to_v2(hass, entry)

    if entry.version == 2:
        await _migrate_v2_to_v3(hass, entry)

    if entry.version == 3:
        await _migrate_v3_to_v4(hass, entry)

    if entry.version == 4:
        await _migrate_v4_to_v5(hass, entry)

    LOGGER.debug(MIGRATION_SUCCESS_VERSION, 5)
    return True


async def _migrate_v1_to_v2(
    hass: HomeAssistant,
    entry: ConfigEntry,
) -> None:
    """Migrate v1: rename 'Name' key to 'Key' in chat_template_kwargs."""
    for subentry in entry.subentries.values():
        if subentry.subentry_type != "conversation":
            continue

        data = dict(subentry.data)
        chat_template_opts = data.get(CONF_CHAT_TEMPLATE_OPTS, {})
        chat_template_kwargs = chat_template_opts.get(CONF_CHAT_TEMPLATE_KWARGS, [])

        migrated = False
        new_kwargs = []
        for item in chat_template_kwargs:
            item_dict = dict(item)
            if "Name" in item_dict and "Key" not in item_dict:
                item_dict["Key"] = item_dict.pop("Name")
                migrated = True
            new_kwargs.append(item_dict)

        if migrated:
            data[CONF_CHAT_TEMPLATE_OPTS] = {
                **chat_template_opts,
                CONF_CHAT_TEMPLATE_KWARGS: new_kwargs,
            }
            hass.config_entries.async_update_subentry(entry, subentry, data=data)

    hass.config_entries.async_update_entry(entry, version=2)


async def _migrate_v2_to_v3(
    hass: HomeAssistant,
    entry: ConfigEntry,
) -> None:
    """Migrate v2: 0/-1 max_message_history → unset (keep-all semantics)."""
    for subentry in entry.subentries.values():
        if subentry.subentry_type != "conversation":
            continue

        data = dict(subentry.data)
        max_message_history = data.get(CONF_MAX_MESSAGE_HISTORY)
        if max_message_history in (0, -1, "0", "-1"):
            del data[CONF_MAX_MESSAGE_HISTORY]
            hass.config_entries.async_update_subentry(entry, subentry, data=data)

    hass.config_entries.async_update_entry(entry, version=3, minor_version=0)


async def _migrate_v3_to_v4(
    hass: HomeAssistant,
    entry: ConfigEntry,
) -> None:
    """Migrate v3: thinking_mode and reasoning_effort conversions."""
    server_type = entry.data.get(CONF_SERVER_TYPE)
    for subentry in entry.subentries.values():
        if subentry.subentry_type not in ("conversation", "ai_task_data"):
            continue

        data = dict(subentry.data)
        migrated = False

        if CONF_LLAMACPP_CONFIG in data:
            llamacpp_config = data[CONF_LLAMACPP_CONFIG]
            if CONF_LLAMACPP_ENABLE_THINKING in llamacpp_config:
                old_enable_thinking = llamacpp_config.pop(CONF_LLAMACPP_ENABLE_THINKING)
                if old_enable_thinking is True:
                    llamacpp_config[CONF_LLAMACPP_THINKING_MODE] = (
                        CONF_LLAMACPP_THINKING_MODE_ALWAYS
                    )
                else:
                    # Preserve the explicit disable — with the option unset the
                    # model's own default behaviour would apply instead.
                    llamacpp_config[CONF_LLAMACPP_THINKING_MODE] = (
                        CONF_LLAMACPP_THINKING_MODE_DISABLED
                    )
                migrated = True

        if CONF_VLLM_CONFIG in data:
            vllm_config = data[CONF_VLLM_CONFIG]
            thinking_token_budget = vllm_config.get(CONF_VLLM_THINKING_TOKEN_BUDGET, 0)
            if int(thinking_token_budget) > 0:
                vllm_config[CONF_VLLM_THINKING_MODE] = CONF_VLLM_THINKING_MODE_ALWAYS
                migrated = True

        if server_type == SERVER_TYPE_DEEPSEEK:
            deepseek_config = data.setdefault(CONF_DEEPSEEK_CONFIG, {})
            old_reasoning_effort = deepseek_config.get(CONF_DEEPSEEK_REASONING_EFFORT)
            if old_reasoning_effort:
                deepseek_config[CONF_DEEPSEEK_THINKING_MODE] = (
                    CONF_DEEPSEEK_THINKING_MODE_ALWAYS
                )
                migrated = True
            if not old_reasoning_effort:
                deepseek_config[CONF_DEEPSEEK_REASONING_EFFORT] = (
                    CONF_DEEPSEEK_REASONING_EFFORT_DEFAULT
                )
                migrated = True

        if migrated:
            hass.config_entries.async_update_subentry(entry, subentry, data=data)

    hass.config_entries.async_update_entry(entry, version=4, minor_version=0)


async def _migrate_v4_to_v5(
    hass: HomeAssistant,
    entry: ConfigEntry,
) -> None:
    """Migrate v4: move content injection fields into context_injection_opts."""
    for subentry in entry.subentries.values():
        if subentry.subentry_type != "conversation":
            continue

        data = dict(subentry.data)
        migrated = False
        context_opts = {}
        if CONF_CONTENT_INJECTION_METHOD in data:
            context_opts[CONF_CONTENT_INJECTION_METHOD] = data.pop(
                CONF_CONTENT_INJECTION_METHOD
            )
            migrated = True
        if CONF_CONTEXT_INJECTION_TEMPLATE in data:
            context_opts[CONF_CONTEXT_INJECTION_TEMPLATE] = data.pop(
                CONF_CONTEXT_INJECTION_TEMPLATE
            )
            migrated = True
        if migrated:
            data[CONF_CONTEXT_INJECTION_OPTS] = context_opts
            hass.config_entries.async_update_subentry(entry, subentry, data=data)

    hass.config_entries.async_update_entry(entry, version=5, minor_version=0)


async def async_setup_entry(hass: HomeAssistant, entry: LocalAiConfigEntry) -> bool:
    """Set up Local OpenAI LLM from a config entry."""
    custom_headers_section = entry.data.get(CONF_CUSTOM_HEADERS, {}) or {}
    custom_headers_list = custom_headers_section.get(CONF_SERVER_HEADERS, []) or []
    custom_headers = {
        item["Key"]: item["Value"] for item in custom_headers_list
    } or None

    client = AsyncOpenAI(
        base_url=entry.data[CONF_BASE_URL],
        api_key=entry.data.get(CONF_API_KEY) or PLACEHOLDER_API_KEY,
        http_client=get_async_client(hass),
        default_headers=custom_headers,
    )

    # Cache current platform data which gets added to each request (caching done by library)
    _ = await hass.async_add_executor_job(client.platform_headers)

    # Access .chat initially to trigger internal openai module loads within an executor
    await hass.async_add_executor_job(lambda: client.chat)

    try:
        async for _ in client.with_options(timeout=10.0).models.list():
            break
    except AuthenticationError as err:
        LOGGER.error("Invalid API key: %s", err)
        msg = "Invalid API key"
        raise ConfigEntryError(msg) from err
    except OpenAIError as err:
        raise ConfigEntryNotReady(err) from err

    entry.runtime_data = client

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    entry.async_on_unload(entry.add_update_listener(_async_update_listener))

    return True


async def _async_update_listener(
    hass: HomeAssistant,
    entry: LocalAiConfigEntry,
) -> None:
    """Handle update."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: LocalAiConfigEntry) -> bool:
    """Unload Local OpenAI LLM."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up is called when Home Assistant is loading our component."""
    service.async_register_platform_entity_service(
        hass=hass,
        service_domain=DOMAIN,
        service_name="add_to_weaviate",
        entity_domain=CONVERSATION_DOMAIN,
        schema={
            vol.Required("query"): str,
            vol.Required("content"): str,
            vol.Optional("identifier"): str,
        },
        func=upsert_data_in_weaviate,
    )
    return True


async def upsert_data_in_weaviate(
    entity: Entity,
    service_call: service.ServiceCall,
) -> None:
    """Service action to add content to Weaviate."""
    await entity.upsert_data_in_weaviate(
        query=service_call.data.get("query"),
        content=service_call.data.get("content"),
        identifier=service_call.data.get("identifier"),
    )
