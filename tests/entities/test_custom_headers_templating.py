"""Tests for Jinja2 template rendering in custom HTTP headers."""

from __future__ import annotations

from types import MappingProxyType
from unittest.mock import MagicMock, patch

import pytest
from custom_components.local_openai.const import (
    CONF_CUSTOM_HEADERS,
    CONF_SERVER_HEADERS,
)
from custom_components.local_openai.conversation import LocalAiConversationEntity
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry, ConfigEntryState, ConfigSubentry
from homeassistant.const import CONF_MODEL
from homeassistant.core import HomeAssistant


@pytest.fixture
async def mock_config_entry_with_headers(
    hass: HomeAssistant,
) -> ConfigEntry:
    """Create a mock config entry with custom headers."""
    entry = ConfigEntry(
        domain="local_openai",
        title="Test Server",
        data={
            CONF_MODEL: "test-model",
            CONF_CUSTOM_HEADERS: {
                CONF_SERVER_HEADERS: [
                    {"Key": "X-Static-Header", "Value": "static-value"},
                    {
                        "Key": "X-Templated-Header",
                        "Value": "{{ states('input_text.session') }}",
                    },
                    {"Key": "X-Number-Template", "Value": "{{ 42 }}"},
                    {
                        "Key": "HTTP-Referer",
                        "Value": "{{ 'https://custom.example.com' }}",
                    },
                ]
            },
        },
        source="user",
        state=ConfigEntryState.LOADED,
        version=1,
        minor_version=1,
        discovery_keys=MappingProxyType({}),
        options=None,
        subentries_data=None,
        unique_id=None,
    )
    entry.runtime_data = MagicMock()
    hass.config_entries._entries[entry.entry_id] = entry
    return entry


@pytest.fixture
async def mock_conversation_subentry_with_headers(
    mock_config_entry_with_headers: ConfigEntry,
) -> ConfigSubentry:
    """Create a mock subentry for conversation agent."""
    subentry = ConfigSubentry(
        subentry_id="test_conversation_subentry_id",
        subentry_type="conversation",
        title="Conversation Agent",
        data=MappingProxyType({CONF_MODEL: "test-model"}),
        unique_id=None,
    )
    mock_config_entry_with_headers._subentries = {
        "test_conversation_subentry_id": subentry,
    }
    return subentry


@pytest.fixture
async def entity_with_headers(
    hass: HomeAssistant,
    mock_config_entry_with_headers: ConfigEntry,
    mock_conversation_subentry_with_headers: ConfigSubentry,
) -> LocalAiConversationEntity:
    """Create a conversation entity with custom headers configured."""
    entity = LocalAiConversationEntity(
        mock_config_entry_with_headers, mock_conversation_subentry_with_headers
    )
    entity.hass = hass
    entity._attr_has_entity_name = False
    entity._attr_name = "Test"
    return entity


def _make_chat_log() -> MagicMock:
    """Create a minimal chat log for testing."""
    chat_log = MagicMock(spec=conversation.ChatLog)
    chat_log.content = []
    chat_log.llm_api = None
    chat_log.conversation_id = "conv-1"
    return chat_log


class TestCustomHeadersJinja2Templating:
    """Test Jinja2 template rendering in custom HTTP headers."""

    async def test_static_headers_passed_through(
        self,
        entity_with_headers: LocalAiConversationEntity,
    ):
        """Static headers (no template brackets) are passed through as-is."""
        captured: dict = {}

        async def fake_loop(client, model_args, *args, **kwargs):
            captured.update(model_args)

        with patch.object(
            entity_with_headers, "_run_agent_loop", side_effect=fake_loop
        ):
            await entity_with_headers._async_handle_chat_log(_make_chat_log())

        extra_headers = captured["extra_headers"]
        assert extra_headers["X-Static-Header"] == "static-value"

    async def test_template_headers_rendered(
        self,
        entity_with_headers: LocalAiConversationEntity,
    ):
        """Templated headers are rendered per-request."""
        captured: dict = {}

        async def fake_loop(client, model_args, *args, **kwargs):
            captured.update(model_args)

        with patch.object(
            entity_with_headers, "_run_agent_loop", side_effect=fake_loop
        ):
            await entity_with_headers._async_handle_chat_log(_make_chat_log())

        extra_headers = captured["extra_headers"]
        assert extra_headers["X-Templated-Header"] == "unknown"
        assert extra_headers["X-Number-Template"] == "42"

    async def test_custom_headers_dont_override_openrouter_headers(
        self,
        entity_with_headers: LocalAiConversationEntity,
    ):
        """Custom headers with the same key as OpenRouter headers are ignored."""
        captured: dict = {}

        async def fake_loop(client, model_args, *args, **kwargs):
            captured.update(model_args)

        with patch.object(
            entity_with_headers, "_run_agent_loop", side_effect=fake_loop
        ):
            await entity_with_headers._async_handle_chat_log(_make_chat_log())

        extra_headers = captured["extra_headers"]
        assert (
            extra_headers["HTTP-Referer"]
            == "https://github.com/skye-harris/hass_local_openai_llm"
        )
        assert extra_headers["X-Title"] == "Home Assistant"

    async def test_mixed_static_and_templated_headers(
        self,
        entity_with_headers: LocalAiConversationEntity,
    ):
        """Static and templated headers coexist in the final extra_headers."""
        captured: dict = {}

        async def fake_loop(client, model_args, *args, **kwargs):
            captured.update(model_args)

        with patch.object(
            entity_with_headers, "_run_agent_loop", side_effect=fake_loop
        ):
            await entity_with_headers._async_handle_chat_log(_make_chat_log())

        extra_headers = captured["extra_headers"]
        assert "X-Static-Header" in extra_headers
        assert "X-Templated-Header" in extra_headers
        assert "X-Number-Template" in extra_headers
        assert "HTTP-Referer" in extra_headers
        assert "X-Title" in extra_headers

    async def test_no_custom_headers_configured(
        self,
        hass: HomeAssistant,
    ):
        """When no custom headers are configured, extra_headers has only OpenRouter headers."""
        entry = ConfigEntry(
            domain="local_openai",
            title="Test Server",
            data={CONF_MODEL: "test-model"},
            source="user",
            state=ConfigEntryState.LOADED,
            version=1,
            minor_version=1,
            discovery_keys=MappingProxyType({}),
            options=None,
            subentries_data=None,
            unique_id=None,
        )
        entry.runtime_data = MagicMock()
        hass.config_entries._entries[entry.entry_id] = entry

        subentry = ConfigSubentry(
            subentry_id="test_subentry_id",
            subentry_type="conversation",
            title="Conversation Agent",
            data=MappingProxyType({CONF_MODEL: "test-model"}),
            unique_id=None,
        )
        entry._subentries = {"test_subentry_id": subentry}

        entity = LocalAiConversationEntity(entry, subentry)
        entity.hass = hass
        entity._attr_has_entity_name = False
        entity._attr_name = "Test"

        captured: dict = {}

        async def fake_loop(client, model_args, *args, **kwargs):
            captured.update(model_args)

        with patch.object(entity, "_run_agent_loop", side_effect=fake_loop):
            await entity._async_handle_chat_log(_make_chat_log())

        extra_headers = captured["extra_headers"]
        assert (
            extra_headers["HTTP-Referer"]
            == "https://github.com/skye-harris/hass_local_openai_llm"
        )
        assert extra_headers["X-Title"] == "Home Assistant"
        assert len(extra_headers) == 2

    async def test_template_render_error_sets_empty_string(
        self,
        hass: HomeAssistant,
    ):
        """A template that raises an error results in an empty string header value."""
        entry = ConfigEntry(
            domain="local_openai",
            title="Test Server",
            data={
                CONF_MODEL: "test-model",
                CONF_CUSTOM_HEADERS: {
                    CONF_SERVER_HEADERS: [
                        {
                            "Key": "X-Bad-Template",
                            "Value": "{{ undefined_var.nonexistent.attr }}",
                        },
                        {"Key": "X-Good-Header", "Value": "good-value"},
                    ]
                },
            },
            source="user",
            state=ConfigEntryState.LOADED,
            version=1,
            minor_version=1,
            discovery_keys=MappingProxyType({}),
            options=None,
            subentries_data=None,
            unique_id=None,
        )
        entry.runtime_data = MagicMock()
        hass.config_entries._entries[entry.entry_id] = entry

        subentry = ConfigSubentry(
            subentry_id="test_subentry_id",
            subentry_type="conversation",
            title="Conversation Agent",
            data=MappingProxyType({CONF_MODEL: "test-model"}),
            unique_id=None,
        )
        entry._subentries = {"test_subentry_id": subentry}

        entity = LocalAiConversationEntity(entry, subentry)
        entity.hass = hass
        entity._attr_has_entity_name = False
        entity._attr_name = "Test"

        captured: dict = {}

        async def fake_loop(client, model_args, *args, **kwargs):
            captured.update(model_args)

        with patch.object(entity, "_run_agent_loop", side_effect=fake_loop):
            await entity._async_handle_chat_log(_make_chat_log())

        extra_headers = captured["extra_headers"]
        assert extra_headers["X-Bad-Template"] == ""
        assert extra_headers["X-Good-Header"] == "good-value"

    async def test_template_with_hass_state(
        self,
        hass: HomeAssistant,
    ):
        """Templates can access Home Assistant state values."""
        entry = ConfigEntry(
            domain="local_openai",
            title="Test Server",
            data={
                CONF_MODEL: "test-model",
                CONF_CUSTOM_HEADERS: {
                    CONF_SERVER_HEADERS: [
                        {
                            "Key": "X-Session-Id",
                            "Value": "{{ states('input_text.session_id') }}",
                        },
                    ]
                },
            },
            source="user",
            state=ConfigEntryState.LOADED,
            version=1,
            minor_version=1,
            discovery_keys=MappingProxyType({}),
            options=None,
            subentries_data=None,
            unique_id=None,
        )
        entry.runtime_data = MagicMock()
        hass.config_entries._entries[entry.entry_id] = entry

        # Create a mock input_text entity with a known state
        hass.states.async_set("input_text.session_id", "my-session-123")

        subentry = ConfigSubentry(
            subentry_id="test_subentry_id",
            subentry_type="conversation",
            title="Conversation Agent",
            data=MappingProxyType({CONF_MODEL: "test-model"}),
            unique_id=None,
        )
        entry._subentries = {"test_subentry_id": subentry}

        entity = LocalAiConversationEntity(entry, subentry)
        entity.hass = hass
        entity._attr_has_entity_name = False
        entity._attr_name = "Test"

        captured: dict = {}

        async def fake_loop(client, model_args, *args, **kwargs):
            captured.update(model_args)

        with patch.object(entity, "_run_agent_loop", side_effect=fake_loop):
            await entity._async_handle_chat_log(_make_chat_log())

        extra_headers = captured["extra_headers"]
        assert extra_headers["X-Session-Id"] == "my-session-123"

    async def test_empty_key_headers_filtered(
        self,
        hass: HomeAssistant,
    ):
        """Headers with empty keys are not added to extra_headers."""
        entry = ConfigEntry(
            domain="local_openai",
            title="Test Server",
            data={
                CONF_MODEL: "test-model",
                CONF_CUSTOM_HEADERS: {
                    CONF_SERVER_HEADERS: [
                        {"Key": "", "Value": "should-be-ignored"},
                        {"Key": " ", "Value": "also-ignored"},
                        {"Key": "X-Valid", "Value": "kept"},
                    ]
                },
            },
            source="user",
            state=ConfigEntryState.LOADED,
            version=1,
            minor_version=1,
            discovery_keys=MappingProxyType({}),
            options=None,
            subentries_data=None,
            unique_id=None,
        )
        entry.runtime_data = MagicMock()
        hass.config_entries._entries[entry.entry_id] = entry

        subentry = ConfigSubentry(
            subentry_id="test_subentry_id",
            subentry_type="conversation",
            title="Conversation Agent",
            data=MappingProxyType({CONF_MODEL: "test-model"}),
            unique_id=None,
        )
        entry._subentries = {"test_subentry_id": subentry}

        entity = LocalAiConversationEntity(entry, subentry)
        entity.hass = hass
        entity._attr_has_entity_name = False
        entity._attr_name = "Test"

        captured: dict = {}

        async def fake_loop(client, model_args, *args, **kwargs):
            captured.update(model_args)

        with patch.object(entity, "_run_agent_loop", side_effect=fake_loop):
            await entity._async_handle_chat_log(_make_chat_log())

        extra_headers = captured["extra_headers"]
        assert "X-Valid" in extra_headers
        assert extra_headers["X-Valid"] == "kept"
