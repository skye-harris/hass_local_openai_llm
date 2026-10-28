"""Tests for the decision model subentry flow."""

from __future__ import annotations

from types import MappingProxyType
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.const import CONF_MODEL
from openai import OpenAIError
from openai.types import Model

from custom_components.local_openai.config_flow import (
    AITaskEvaluateFlowHandler,
    LocalAiConfigFlow,
    _is_decision_model,
)
from custom_components.local_openai.const import (
    CONF_SERVER_TYPE,
    SERVER_TYPE_DEEPSEEK,
    SERVER_TYPE_GENERIC,
    SERVER_TYPE_GOOGLE_GEMINI,
    SERVER_TYPE_LLAMACPP,
    SERVER_TYPE_LOCALAI,
    SERVER_TYPE_VLLM,
)


def _model(model_id: str, output_modalities: list[str] | None = None) -> Model:
    """Create a /v1/models entry, with llama.cpp architecture metadata if given."""
    extra = (
        {}
        if output_modalities is None
        else {
            "architecture": {
                "input_modalities": ["text"],
                "output_modalities": output_modalities,
            },
        }
    )
    return Model(id=model_id, created=0, object="model", owned_by="llamacpp", **extra)


def _handler(server_type: str, models: list[Model]) -> AITaskEvaluateFlowHandler:
    """Create a flow handler for a server listing the given models."""
    entry = MagicMock()
    entry.title = "Test Server"
    entry.data = {CONF_SERVER_TYPE: server_type}
    entry.runtime_data.models.list = AsyncMock(return_value=MagicMock(data=models))
    handler = AITaskEvaluateFlowHandler()
    handler._get_entry = MagicMock(return_value=entry)
    return handler


def _model_options(schema) -> list[str]:
    """Return the model picker values of a schema."""
    selector = next(
        value for marker, value in schema.schema.items() if str(marker) == CONF_MODEL
    )
    return [option["value"] for option in selector.config["options"]]


class TestIsDecisionModel:
    """Tests for _is_decision_model."""

    @pytest.mark.parametrize(
        ("output_modalities", "expected"),
        [
            (None, True),
            (["decisions"], True),
            (["text", "decisions"], True),
            (["text"], False),
            (["audio"], False),
        ],
    )
    def test_output_modalities(
        self,
        output_modalities: list[str] | None,
        expected: bool,
    ) -> None:
        """Test models are filtered on the decisions output modality."""
        assert _is_decision_model(_model("model", output_modalities)) is expected


class TestSupportedSubentryTypes:
    """Tests for which servers offer decision model subentries."""

    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            ({}, True),
            ({CONF_SERVER_TYPE: SERVER_TYPE_GENERIC}, True),
            ({CONF_SERVER_TYPE: SERVER_TYPE_LLAMACPP}, True),
            ({CONF_SERVER_TYPE: SERVER_TYPE_VLLM}, False),
            ({CONF_SERVER_TYPE: SERVER_TYPE_LOCALAI}, False),
            ({CONF_SERVER_TYPE: SERVER_TYPE_DEEPSEEK}, False),
            ({CONF_SERVER_TYPE: SERVER_TYPE_GOOGLE_GEMINI}, False),
        ],
    )
    def test_decision_subentry_offered(self, data: dict, expected: bool) -> None:
        """Test decision models are only offered for servers with /v1/systemone."""
        entry = MagicMock()
        entry.data = data
        subentry_types = LocalAiConfigFlow.async_get_supported_subentry_types(entry)
        assert ("ai_task_evaluate" in subentry_types) is expected
        assert {"conversation", "ai_task_data"} <= set(subentry_types)


class TestAITaskEvaluateSchema:
    """Tests for the decision model subentry form."""

    async def test_lists_only_decision_models(self) -> None:
        """Test models llama.cpp reports as non-decision models are hidden."""
        handler = _handler(
            SERVER_TYPE_LLAMACPP,
            [_model("qwen3-8b", ["text"]), _model("openjev", ["decisions"])],
        )
        assert _model_options(await handler.get_schema()) == ["openjev"]

    async def test_lists_all_models_without_metadata(self) -> None:
        """Test all models are listed when the server gives no architecture."""
        handler = _handler(SERVER_TYPE_GENERIC, [_model("nimble"), _model("clef")])
        assert _model_options(await handler.get_schema()) == ["nimble", "clef"]

    async def test_models_list_error(self) -> None:
        """Test a failure listing models leaves the picker empty."""
        handler = _handler(SERVER_TYPE_GENERIC, [])
        handler._get_entry().runtime_data.models.list.side_effect = OpenAIError()
        assert _model_options(await handler.get_schema()) == []


class TestAITaskEvaluateSteps:
    """Tests for the decision model subentry steps."""

    async def test_user_step_creates_entry(self) -> None:
        """Test the subentry is titled after the server and model."""
        handler = _handler(SERVER_TYPE_LLAMACPP, [])
        handler.async_create_entry = MagicMock()

        await handler.async_step_user({CONF_MODEL: "/models/openjev.gguf"})

        handler.async_create_entry.assert_called_once_with(
            title="Test Server: openjev Decisions",
            data={CONF_MODEL: "/models/openjev.gguf"},
        )

    async def test_reconfigure_step_updates_subentry(self) -> None:
        """Test reconfiguring replaces the subentry data."""
        handler = _handler(SERVER_TYPE_LLAMACPP, [])
        subentry = MagicMock(data=MappingProxyType({CONF_MODEL: "openjev"}))
        handler._get_reconfigure_subentry = MagicMock(return_value=subentry)
        handler.async_update_and_abort = MagicMock()

        await handler.async_step_reconfigure({CONF_MODEL: "clef"})

        handler.async_update_and_abort.assert_called_once_with(
            entry=handler._get_entry(),
            subentry=subentry,
            data={CONF_MODEL: "clef"},
        )
