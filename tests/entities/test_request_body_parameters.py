"""Tests for custom request body parameters."""

from __future__ import annotations

from types import MappingProxyType
from unittest.mock import MagicMock

from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_MODEL
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.local_openai.const import (
    CONF_CHAT_TEMPLATE_KWARGS,
    CONF_CHAT_TEMPLATE_OPTS,
    CONF_LLAMACPP_CONFIG,
    CONF_LLAMACPP_TOP_P,
    CONF_PASS_SESSION_ID,
    CONF_REQUEST_BODY_OPTS,
    CONF_REQUEST_BODY_PARAMETERS,
    CONF_SERVER_OPTIONS,
    CONF_SERVER_TYPE,
    SERVER_TYPE_GENERIC,
    SERVER_TYPE_LLAMACPP,
)
from custom_components.local_openai.conversation import LocalAiConversationEntity
from custom_components.local_openai.entities.llama_cpp import LlamaCppConversationEntity
from custom_components.local_openai.entity import AgentContext


def _request_body_options(*parameters: dict[str, str]) -> dict:
    """Return subentry options with request body parameters."""
    return {
        CONF_REQUEST_BODY_OPTS: {
            CONF_REQUEST_BODY_PARAMETERS: list(parameters),
        },
    }


def _request_body_parameter(key: str, value: str = "value") -> dict[str, str]:
    """Return a request body parameter entry."""
    return {"Key": key, "Value": value}


def _make_entry(
    server_type: str = SERVER_TYPE_GENERIC,
    server_options: dict | None = None,
) -> MockConfigEntry:
    """Create a config entry for a server type."""
    data = {CONF_SERVER_TYPE: server_type}
    if server_options is not None:
        data[CONF_SERVER_OPTIONS] = server_options
    entry = MockConfigEntry(
        domain="local_openai",
        title="Test Server",
        data=data,
        source="user",
        state="loaded",
        version=1,
        minor_version=1,
        discovery_keys=MappingProxyType({}),
        unique_id=None,
    )
    entry.runtime_data = MagicMock()
    return entry


def _make_subentry(data: dict) -> ConfigSubentry:
    """Create a conversation subentry."""
    return ConfigSubentry(
        subentry_id="test_subentry_id",
        subentry_type="conversation",
        title="Test",
        data=MappingProxyType(data),
        unique_id=None,
    )


def test_request_body_parameters_render_template_values(
    hass: HomeAssistant,
):
    """Test request body parameter values are rendered like templates."""
    subentry = _make_subentry(
        {
            CONF_MODEL: "test-model",
            **_request_body_options(
                _request_body_parameter("reasoning_effort", "medium"),
                _request_body_parameter("max_tokens", "{{ 8192 }}"),
                _request_body_parameter("include_reasoning", "{{ true }}"),
                _request_body_parameter("", "ignored"),
            ),
        },
    )
    entity = LocalAiConversationEntity(_make_entry(), subentry)
    entity.hass = hass

    context = AgentContext()
    entity._get_extra_body_args(context)

    assert context.extra_body == {
        "stream_options": {"include_usage": True},
        "reasoning_effort": "medium",
        "max_tokens": 8192,
        "include_reasoning": True,
    }


def test_request_body_parameters_are_merged_with_chat_template_kwargs(
    hass: HomeAssistant,
):
    """Test request body parameters are merged with chat template kwargs."""
    subentry = _make_subentry(
        {
            CONF_MODEL: "test-model",
            CONF_CHAT_TEMPLATE_OPTS: {
                CONF_CHAT_TEMPLATE_KWARGS: [
                    {"Key": "template_arg", "Value": "{{ 1 }}"},
                ],
            },
            **_request_body_options(
                _request_body_parameter("reasoning_effort", "medium"),
            ),
        },
    )
    entity = LocalAiConversationEntity(_make_entry(), subentry)
    entity.hass = hass

    context = AgentContext()
    entity._get_extra_body_args(context)

    assert context.extra_body == {
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"template_arg": 1},
        "reasoning_effort": "medium",
    }


def test_request_body_parameters_are_merged_after_server_specific_args(
    hass: HomeAssistant,
):
    """Test custom request body parameters apply to server-specific entities."""
    entry = _make_entry(SERVER_TYPE_LLAMACPP)
    subentry = _make_subentry(
        {
            CONF_MODEL: "test-model",
            CONF_LLAMACPP_CONFIG: {
                CONF_LLAMACPP_TOP_P: 0.8,
            },
            **_request_body_options(_request_body_parameter("seed", "{{ 123 }}")),
        },
    )
    entry._subentries = {"test_subentry_id": subentry}
    entity = LlamaCppConversationEntity(entry, subentry)
    entity.hass = hass

    context = AgentContext()
    entity._get_extra_body_args(context)

    assert context.extra_body == {
        "chat_template_kwargs": {"enable_thinking": False},
        "top_p": 0.8,
        "seed": 123,
    }


def test_session_id_added_to_metadata_when_enabled(hass: HomeAssistant):
    """Test the conversation ID is passed as metadata.session_id when enabled."""
    entry = _make_entry(server_options={CONF_PASS_SESSION_ID: True})
    subentry = _make_subentry({CONF_MODEL: "test-model"})
    entity = LocalAiConversationEntity(entry, subentry)
    entity.hass = hass

    context = AgentContext(conversation_id="test_conversation_id")
    entity._get_extra_body_args(context)

    assert context.extra_body["metadata"]["session_id"] == "test_conversation_id"


def test_session_id_omitted_when_disabled_or_missing(hass: HomeAssistant):
    """Test metadata.session_id is absent without the option or conversation ID."""
    entry = _make_entry(server_options={CONF_PASS_SESSION_ID: False})
    subentry = _make_subentry({CONF_MODEL: "test-model"})
    entity = LocalAiConversationEntity(entry, subentry)
    entity.hass = hass

    disabled = AgentContext(conversation_id="test_conversation_id")
    entity._get_extra_body_args(disabled)
    no_conversation = AgentContext()
    entity._get_extra_body_args(no_conversation)

    assert "metadata" not in disabled.extra_body
    assert "metadata" not in no_conversation.extra_body
