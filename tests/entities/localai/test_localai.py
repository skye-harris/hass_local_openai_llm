"""Unit tests for LocalAI server-specific entities."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_MODEL
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry
from types import MappingProxyType

from custom_components.local_openai.ai_task import LocalAITaskEntity
from custom_components.local_openai.const import (
    CONF_CHAT_TEMPLATE_KWARGS,
    CONF_CHAT_TEMPLATE_OPTS,
)
from custom_components.local_openai.conversation import LocalAiConversationEntity
from custom_components.local_openai.entities.localai import (
    LocalAIServerAITaskEntity,
    LocalAIServerConversationEntity,
    _to_metadata_value,
)


class TestToMetadataValue:
    """Tests for _to_metadata_value helper function."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (True, "true"),
            (False, "false"),
            (42, "42"),
            (0, "0"),
            (-1, "-1"),
            (3.14, "3.14"),
            (0.0, "0.0"),
            ("hello", "hello"),
            ("", ""),
            ("True", "True"),
            ("False", "False"),
        ],
    )
    def test_to_metadata_value(self, value: Any, expected: str):
        """Test value conversion to metadata string format."""
        assert _to_metadata_value(value) == expected


class TestLocalAIServerMixinExtraBodyArgs:
    """Tests for LocalAIServerMixin._get_extra_body_args."""

    def _make_entity(
        self,
        hass: HomeAssistant,
        subentry_data: dict,
    ) -> LocalAIServerConversationEntity:
        entry = MockConfigEntry(
            domain="local_openai",
            title="Test Server",
            data={},
            source="user",
            state="loaded",
            version=1,
            minor_version=1,
            discovery_keys=MappingProxyType({}),
            unique_id=None,
        )
        entry.runtime_data = MagicMock()
        subentry = ConfigSubentry(
            subentry_id="test_subentry_id",
            subentry_type="conversation",
            title="Test",
            data=MappingProxyType(subentry_data),
            unique_id=None,
        )
        entry._subentries = {"test_subentry_id": subentry}
        entity = LocalAIServerConversationEntity(entry, subentry)
        entity.hass = hass
        return entity

    def test_kwargs_routed_to_metadata(self, hass: HomeAssistant):
        """Test chat_template_kwargs are routed into metadata field."""
        entity = self._make_entity(
            hass,
            {
                CONF_MODEL: "test-model",
                CONF_CHAT_TEMPLATE_OPTS: {
                    CONF_CHAT_TEMPLATE_KWARGS: [
                        {"Key": "enable_thinking", "Value": "true"},
                    ]
                },
            },
        )
        result = entity._get_extra_body_args(
            MappingProxyType(
                {
                    CONF_CHAT_TEMPLATE_OPTS: {
                        "chat_template_kwargs": [
                            {"Key": "enable_thinking", "Value": "true"}
                        ]
                    }
                }
            ),
        )
        assert "metadata" in result
        assert result["metadata"]["enable_thinking"] == "true"
        assert "chat_template_kwargs" not in result

    @pytest.mark.parametrize(
        "options",
        [
            MappingProxyType({CONF_CHAT_TEMPLATE_OPTS: {"chat_template_kwargs": []}}),
            MappingProxyType({}),
        ],
    )
    def test_no_kwargs_no_metadata(
        self,
        hass: HomeAssistant,
        options: MappingProxyType,
    ):
        """Test that absent chat_template_kwargs does not create metadata."""
        entity = self._make_entity(
            hass,
            {CONF_MODEL: "test-model"},
        )
        result = entity._get_extra_body_args(options)
        assert "metadata" not in result

    def test_boolean_kwargs_lowercase(self, hass: HomeAssistant):
        """Test boolean values are converted to lowercase strings."""
        entity = self._make_entity(
            hass,
            {CONF_MODEL: "test-model"},
        )
        with patch.object(
            LocalAiConversationEntity,
            "_get_extra_body_args",
            return_value={
                "chat_template_kwargs": {"enable_thinking": True, "disabled": False}
            },
        ):
            result = entity._get_extra_body_args(MappingProxyType({}))
        assert result["metadata"]["enable_thinking"] == "true"
        assert result["metadata"]["disabled"] == "false"

    def test_preserves_parent_extra_body_args(self, hass: HomeAssistant):
        """Test that parent extra_body_args are preserved."""
        entity = self._make_entity(
            hass,
            {CONF_MODEL: "test-model"},
        )
        with patch.object(
            LocalAiConversationEntity,
            "_get_extra_body_args",
            return_value={"chat_template_kwargs": {"enable_thinking": False}},
        ):
            result = entity._get_extra_body_args(MappingProxyType({}))
        assert result["metadata"]["enable_thinking"] == "false"
        assert "chat_template_kwargs" not in result


class TestLocalAIServerConversationEntity:
    """Tests for LocalAIServerConversationEntity."""

    def test_entity_properties(self, hass: HomeAssistant):
        """Test entity inherits from LocalAiConversationEntity and has mixin methods."""
        entry = MockConfigEntry(
            domain="local_openai",
            title="Test Server",
            data={},
            source="user",
            state="loaded",
            version=1,
            minor_version=1,
            discovery_keys=MappingProxyType({}),
            unique_id=None,
        )
        entry.runtime_data = MagicMock()
        subentry = ConfigSubentry(
            subentry_id="test_subentry_id",
            subentry_type="conversation",
            title="Test",
            data=MappingProxyType({CONF_MODEL: "test-model"}),
            unique_id=None,
        )
        entry._subentries = {"test_subentry_id": subentry}
        entity = LocalAIServerConversationEntity(entry, subentry)
        entity.hass = hass
        assert isinstance(entity, LocalAiConversationEntity)
        assert hasattr(entity, "_get_extra_body_args")


class TestLocalAIServerAITaskEntity:
    """Tests for LocalAIServerAITaskEntity."""

    def test_entity_properties(self, hass: HomeAssistant):
        """Test entity inherits from LocalAITaskEntity and has mixin methods."""
        entry = MockConfigEntry(
            domain="local_openai",
            title="Test Server",
            data={},
            source="user",
            state="loaded",
            version=1,
            minor_version=1,
            discovery_keys=MappingProxyType({}),
            unique_id=None,
        )
        entry.runtime_data = MagicMock()
        subentry = ConfigSubentry(
            subentry_id="test_subentry_id",
            subentry_type="ai_task_data",
            title="Test",
            data=MappingProxyType({CONF_MODEL: "test-model"}),
            unique_id=None,
        )
        entry._subentries = {"test_subentry_id": subentry}
        entity = LocalAIServerAITaskEntity(entry, subentry)
        entity.hass = hass
        assert isinstance(entity, LocalAITaskEntity)
        assert hasattr(entity, "_get_extra_body_args")
