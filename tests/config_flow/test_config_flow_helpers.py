"""Unit tests for config_flow helper functions."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import voluptuous as vol
from homeassistant.const import CONF_MODEL

from custom_components.local_openai.config_flow import (
    AI_TASK_SCHEMA_PROVIDERS,
    AITaskDataFlowHandler,
    CONVERSATION_SCHEMA_PROVIDERS,
    REQUEST_BODY_PARAMETER_ALREADY_CONFIGURABLE,
    REQUEST_BODY_PARAMETER_RESERVED,
    _get_ai_task_config_schema,
    _get_conversation_config_schema,
    _get_request_body_parameter_error,
    _get_server_type_config_key,
    _key_value_template_section,
    _resolve_model_name,
    options_to_selections_dict,
    prepare_weaviate_class,
)
from custom_components.local_openai.const import (
    CONF_CHAT_TEMPLATE_KWARGS,
    CONF_DEEPSEEK_CONFIG,
    CONF_GENERIC_CONFIG,
    CONF_LLAMACPP_CONFIG,
    CONF_LOCALAI_CONFIG,
    CONF_REQUEST_BODY_OPTS,
    CONF_REQUEST_BODY_PARAMETERS,
    CONF_TEMPERATURE,
    CONF_VLLM_CONFIG,
    SERVER_TYPE_DEEPSEEK,
    SERVER_TYPE_GENERIC,
    SERVER_TYPE_GOOGLE_GEMINI,
    SERVER_TYPE_LLAMACPP,
    SERVER_TYPE_LOCALAI,
    SERVER_TYPE_VLLM,
)


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


class TestKeyValueTemplateSection:
    """Tests for _key_value_template_section."""

    def test_key_value_template_section(self) -> None:
        """Test key/value template section validation."""
        schema = _key_value_template_section(CONF_REQUEST_BODY_PARAMETERS)

        assert schema({}) == {CONF_REQUEST_BODY_PARAMETERS: []}


class TestRequestBodyParameterError:
    """Tests for _get_request_body_parameter_error."""

    @pytest.mark.parametrize(
        ("key", "expected_error"),
        [
            pytest.param("messages", REQUEST_BODY_PARAMETER_RESERVED, id="messages"),
            pytest.param("metadata", REQUEST_BODY_PARAMETER_RESERVED, id="metadata"),
            pytest.param(
                CONF_MODEL,
                REQUEST_BODY_PARAMETER_ALREADY_CONFIGURABLE,
                id="model",
            ),
            pytest.param(
                CONF_TEMPERATURE,
                REQUEST_BODY_PARAMETER_ALREADY_CONFIGURABLE,
                id="temperature",
            ),
            pytest.param(
                CONF_CHAT_TEMPLATE_KWARGS,
                REQUEST_BODY_PARAMETER_ALREADY_CONFIGURABLE,
                id="chat_template_kwargs",
            ),
        ],
    )
    def test_global_errors(self, key: str, expected_error: str) -> None:
        """Test global request body parameter denylist errors."""
        assert _get_request_body_parameter_error(
            _request_body_options(_request_body_parameter(key)),
            SERVER_TYPE_GENERIC,
        ) == (expected_error, key)

    @pytest.mark.parametrize(
        ("server_type", "key", "expected_error"),
        [
            pytest.param(
                SERVER_TYPE_LLAMACPP,
                "top_p",
                REQUEST_BODY_PARAMETER_ALREADY_CONFIGURABLE,
                id="llamacpp_top_p",
            ),
            pytest.param(
                SERVER_TYPE_LLAMACPP,
                "id_slot",
                REQUEST_BODY_PARAMETER_ALREADY_CONFIGURABLE,
                id="llamacpp_id_slot",
            ),
            pytest.param(
                SERVER_TYPE_DEEPSEEK,
                "reasoning_effort",
                REQUEST_BODY_PARAMETER_ALREADY_CONFIGURABLE,
                id="deepseek_reasoning_effort",
            ),
            pytest.param(
                SERVER_TYPE_DEEPSEEK,
                "thinking",
                REQUEST_BODY_PARAMETER_RESERVED,
                id="deepseek_thinking",
            ),
            pytest.param(
                SERVER_TYPE_VLLM,
                "thinking_token_budget",
                REQUEST_BODY_PARAMETER_ALREADY_CONFIGURABLE,
                id="vllm_thinking_token_budget",
            ),
        ],
    )
    def test_server_type_errors(
        self,
        server_type: str,
        key: str,
        expected_error: str,
    ) -> None:
        """Test server-specific request body parameter denylist errors."""
        assert _get_request_body_parameter_error(
            _request_body_options(_request_body_parameter(key)),
            server_type,
        ) == (expected_error, key)

    def test_generic_server_allows_provider_specific_parameters(self) -> None:
        """Test generic servers can use provider-specific parameters."""
        assert (
            _get_request_body_parameter_error(
                _request_body_options(
                    _request_body_parameter("reasoning_effort"),
                    _request_body_parameter("top_p"),
                    _request_body_parameter("max_tokens"),
                ),
                SERVER_TYPE_GENERIC,
            )
            is None
        )


class TestGetServerTypeConfigKey:
    """Tests for _get_server_type_config_key."""

    @pytest.mark.parametrize(
        ("server_type", "expected_key"),
        [
            pytest.param(SERVER_TYPE_GENERIC, CONF_GENERIC_CONFIG, id="generic"),
            pytest.param(SERVER_TYPE_LLAMACPP, CONF_LLAMACPP_CONFIG, id="llamacpp"),
            pytest.param(SERVER_TYPE_VLLM, CONF_VLLM_CONFIG, id="vllm"),
            pytest.param(SERVER_TYPE_DEEPSEEK, CONF_DEEPSEEK_CONFIG, id="deepseek"),
            pytest.param(SERVER_TYPE_LOCALAI, CONF_LOCALAI_CONFIG, id="localai"),
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
    """Ensure SERVER_TYPE_TO_CONFIG_KEY has an entry for each entity file plus generic."""

    def test_server_type_count_matches_entity_py_files(self) -> None:
        """Assert the number of server types equals entity py files plus the generic fallback."""
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
        assert len(SERVER_TYPE_TO_CONFIG_KEY) == len(entity_files) + 1, (
            f"SERVER_TYPE_TO_CONFIG_KEY has {len(SERVER_TYPE_TO_CONFIG_KEY)} entries but "
            f"entities/ has {len(entity_files)} py files (excluding __init__.py); "
            "expected entity count + 1 for the generic fallback"
        )


class TestGetConversationConfigSchema:
    """Tests for _get_conversation_config_schema."""

    @pytest.mark.parametrize(
        "server_type",
        [
            pytest.param(server, id=server)
            for server in CONVERSATION_SCHEMA_PROVIDERS
            if server not in (SERVER_TYPE_GOOGLE_GEMINI, SERVER_TYPE_LOCALAI)
        ],
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

    def test_google_gemini_returns_empty(self) -> None:
        """Test google_gemini server type returns an empty dict."""
        result = _get_conversation_config_schema(SERVER_TYPE_GOOGLE_GEMINI)
        assert result == {}

    def test_localai_returns_empty(self) -> None:
        """Test localai server type returns an empty dict."""
        result = _get_conversation_config_schema(SERVER_TYPE_LOCALAI)
        assert result == {}


class TestGetAiTaskConfigSchema:
    """Tests for _get_ai_task_config_schema."""

    @pytest.mark.parametrize(
        "server_type",
        [
            pytest.param(server, id=server)
            for server in AI_TASK_SCHEMA_PROVIDERS
            if server not in (SERVER_TYPE_GOOGLE_GEMINI, SERVER_TYPE_LOCALAI)
        ],
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

    def test_google_gemini_returns_empty(self) -> None:
        """Test google_gemini server type returns an empty dict."""
        result = _get_ai_task_config_schema(SERVER_TYPE_GOOGLE_GEMINI)
        assert result == {}

    def test_localai_returns_empty(self) -> None:
        """Test localai server type returns an empty dict."""
        result = _get_ai_task_config_schema(SERVER_TYPE_LOCALAI)
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


def _marker_for(schema: vol.Schema, key: str):
    """Return the voluptuous Marker whose schema string equals key."""
    for marker in schema.schema:
        if str(marker) == key:
            return marker
    return None


class TestAITaskSchemaTemperature:
    """Tests for AI Task schema temperature field."""

    async def test_ai_task_schema_has_optional_temperature(self, hass) -> None:
        """Test that AI Task schema has an optional temperature field."""
        handler = AITaskDataFlowHandler()
        handler.hass = hass

        entry = MagicMock()
        entry.data = {}
        entry.runtime_data = MagicMock()
        entry.runtime_data.models.list = AsyncMock(return_value=MagicMock(data=[]))
        handler._get_entry = MagicMock(return_value=entry)

        schema = await handler.get_schema()

        marker = _marker_for(schema, CONF_TEMPERATURE)
        assert marker is not None, "AI Task schema must expose a temperature field"
        assert isinstance(marker, vol.Optional)
        # Optional with no forced default → unset stays unset.
        assert marker.default is vol.UNDEFINED
