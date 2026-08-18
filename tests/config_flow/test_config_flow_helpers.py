"""Unit tests for config_flow helper functions."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from custom_components.local_openai.config_flow import (
    AI_TASK_SCHEMA_PROVIDERS,
    CONVERSATION_SCHEMA_PROVIDERS,
    _get_ai_task_config_schema,
    _get_conversation_config_schema,
    _get_server_type_config_key,
    _resolve_model_name,
    options_to_selections_dict,
    prepare_weaviate_class,
)
from custom_components.local_openai.const import (
    CONF_DEEPSEEK_CONFIG,
    CONF_GENERIC_CONFIG,
    CONF_LLAMACPP_CONFIG,
    CONF_VLLM_CONFIG,
    SERVER_TYPE_DEEPSEEK,
    SERVER_TYPE_GENERIC,
    SERVER_TYPE_LLAMACPP,
    SERVER_TYPE_VLLM,
)


class TestOptionsToSelectionsDict:
    """Tests for options_to_selections_dict."""

    @pytest.mark.parametrize(
        ("input_dict", "expected"),
        [
            pytest.param(
                {"a": "Alpha", "b": "Beta"},
                [
                    {"value": "a", "label": "Alpha"},
                    {"value": "b", "label": "Beta"},
                ],
                id="two_entries",
            ),
            pytest.param(
                {"single": "One"},
                [{"value": "single", "label": "One"}],
                id="single_entry",
            ),
            pytest.param({}, [], id="empty_dict"),
        ],
    )
    def test_options_to_selections_dict(
        self,
        input_dict: dict,
        expected: list[dict],
    ) -> None:
        """Test conversion of dict to SelectOptionDict list preserves order and values."""
        result = options_to_selections_dict(input_dict)
        assert len(result) == len(expected)
        for item, exp in zip(result, expected):
            assert item["value"] == exp["value"]
            assert item["label"] == exp["label"]


class TestGetServerTypeConfigKey:
    """Tests for _get_server_type_config_key."""

    @pytest.mark.parametrize(
        ("server_type", "expected_key"),
        [
            pytest.param(SERVER_TYPE_GENERIC, CONF_GENERIC_CONFIG, id="generic"),
            pytest.param(SERVER_TYPE_LLAMACPP, CONF_LLAMACPP_CONFIG, id="llamacpp"),
            pytest.param(SERVER_TYPE_VLLM, CONF_VLLM_CONFIG, id="vllm"),
            pytest.param(SERVER_TYPE_DEEPSEEK, CONF_DEEPSEEK_CONFIG, id="deepseek"),
            pytest.param(
                "unknown_type", CONF_GENERIC_CONFIG, id="unknown_defaults_to_generic"
            ),
            pytest.param("", CONF_GENERIC_CONFIG, id="empty_defaults_to_generic"),
        ],
    )
    def test_server_type_config_key(
        self,
        server_type: str,
        expected_key: str,
    ) -> None:
        """Test server type to config key mapping including defaults."""
        assert _get_server_type_config_key(server_type) == expected_key


class TestServerTypeCountMatchesEntityFiles:
    """Ensure SERVER_TYPE_TO_CONFIG_KEY has an entry for each entity file."""

    def test_server_type_count_matches_entity_py_files(self) -> None:
        """Assert the number of server types matches the number of entity py files (excluding __init__.py)."""
        import os
        from pathlib import Path

        from custom_components.local_openai.config_flow import SERVER_TYPE_TO_CONFIG_KEY

        entities_dir = (
            Path(__file__).parent.parent.parent
            / "custom_components"
            / "local_openai"
            / "entities"
        )
        entity_files = [
            file
            for file in os.listdir(entities_dir)
            if file.endswith(".py") and file != "__init__.py"
        ]
        assert len(SERVER_TYPE_TO_CONFIG_KEY) == len(entity_files), (
            f"SERVER_TYPE_TO_CONFIG_KEY has {len(SERVER_TYPE_TO_CONFIG_KEY)} entries but "
            f"entities/ has {len(entity_files)} py files (excluding __init__.py)"
        )


class TestGetConversationConfigSchema:
    """Tests for _get_conversation_config_schema."""

    @pytest.mark.parametrize(
        "server_type",
        [pytest.param(server, id=server) for server in CONVERSATION_SCHEMA_PROVIDERS],
    )
    def test_provider_types_return_non_empty_schema(self, server_type: str) -> None:
        """Test each provider server type returns a non-empty schema dict."""
        result = _get_conversation_config_schema(server_type)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_generic_returns_empty(self) -> None:
        """Test generic server type returns an empty dict."""
        result = _get_conversation_config_schema(SERVER_TYPE_GENERIC)
        assert result == {}


class TestGetAiTaskConfigSchema:
    """Tests for _get_ai_task_config_schema."""

    @pytest.mark.parametrize(
        "server_type",
        [pytest.param(server, id=server) for server in AI_TASK_SCHEMA_PROVIDERS],
    )
    def test_provider_types_return_non_empty_schema(self, server_type: str) -> None:
        """Test each provider server type returns a non-empty schema dict."""
        result = _get_ai_task_config_schema(server_type)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_generic_returns_empty(self) -> None:
        """Test generic server type returns an empty dict."""
        result = _get_ai_task_config_schema(SERVER_TYPE_GENERIC)
        assert result == {}


class TestPrepareWeaviateClass:
    """Tests for prepare_weaviate_class."""

    async def test_no_host_returns_early(self, hass: MagicMock) -> None:
        """Test that missing host returns without creating WeaviateClient."""
        weaviate_opts = {}
        await prepare_weaviate_class(hass, weaviate_opts)

    async def test_class_exists_returns_early(self, hass: MagicMock) -> None:
        """Test that existing class returns without creating."""
        weaviate_opts = {"weaviate_host": "http://localhost:8080"}
        mock_weaviate = AsyncMock()
        mock_weaviate.does_class_exist.return_value = True
        with patch(
            "custom_components.local_openai.config_flow.WeaviateClient",
            return_value=mock_weaviate,
        ):
            await prepare_weaviate_class(hass, weaviate_opts)
        mock_weaviate.create_class.assert_not_called()

    async def test_class_not_exists_creates(self, hass: MagicMock) -> None:
        """Test that non-existing class triggers creation."""
        weaviate_opts = {
            "weaviate_host": "http://localhost:8080",
            "weaviate_class_name": "MyCustomClass",
        }
        mock_weaviate = AsyncMock()
        mock_weaviate.does_class_exist.return_value = False
        with patch(
            "custom_components.local_openai.config_flow.WeaviateClient",
            return_value=mock_weaviate,
        ):
            await prepare_weaviate_class(hass, weaviate_opts)
        mock_weaviate.create_class.assert_called_once_with("MyCustomClass")

    async def test_default_class_name(self, hass: MagicMock) -> None:
        """Test that default class name is used when not provided."""
        weaviate_opts = {"weaviate_host": "http://localhost:8080"}
        mock_weaviate = AsyncMock()
        mock_weaviate.does_class_exist.return_value = False
        with patch(
            "custom_components.local_openai.config_flow.WeaviateClient",
            return_value=mock_weaviate,
        ):
            await prepare_weaviate_class(hass, weaviate_opts)
        mock_weaviate.create_class.assert_called_once_with("Homeassistant")


class TestResolveModelName:
    """Tests for _resolve_model_name."""

    def test_generic_strips_path_and_gguf(self) -> None:
        """Test generic server type strips path and .gguf extension."""
        model = MagicMock(id="models/my-model.gguf")
        assert _resolve_model_name(SERVER_TYPE_GENERIC, model) == "my-model"

    def test_mixin_resolver_called(self) -> None:
        """Test that a mixin (llama.cpp in this case) is used when available."""
        model = MagicMock(id="models/base.gguf")
        with patch(
            "custom_components.local_openai.config_flow._llama_cpp_model_alias",
            return_value="Base Model",
        ) as mock_resolver:
            result = _resolve_model_name(SERVER_TYPE_LLAMACPP, model)
            mock_resolver.assert_called_once_with(model)
            assert result == "Base Model"
