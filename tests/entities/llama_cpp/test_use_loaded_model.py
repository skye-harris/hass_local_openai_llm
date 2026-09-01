"""Tests for the llama.cpp use_loaded_model feature."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from custom_components.local_openai.const import (
    CONF_LLAMACPP_CONFIG,
    CONF_LLAMACPP_USE_LOADED_MODEL,
)
from custom_components.local_openai.entities.llama_cpp import LlamaCppMixin


def _make_chat_log(
    attachments: list[list[str]] | None = None,
) -> MagicMock:
    """Create a mock ChatLog with content items that have optional image attachments.

    Args:
        attachments: List of attachment mime_types per content item.
                     None = no attachments on that content item.
                     Empty list = content item exists but has no attachments.
    """
    content_items = []
    if attachments is None:
        attachments = [None]

    for mime_types in attachments:
        item = MagicMock()
        if mime_types is None:
            item.attachments = None
        elif len(mime_types) == 0:
            item.attachments = []
        else:
            item.attachments = [MagicMock(mime_type=mt) for mt in mime_types]
        content_items.append(item)

    chat_log = MagicMock()
    chat_log.content = content_items
    return chat_log


class _StubSubentry:
    """Stub subentry for testing."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data


class _StubEntry:
    """Stub config entry for testing."""

    def __init__(self, runtime_data: MagicMock) -> None:
        self.runtime_data = runtime_data


class _StubBaseEntity:
    """Stub base entity that mimics LocalAiEntity initialization."""

    def __init__(self, entry: MagicMock, subentry: MagicMock) -> None:
        self.entry = entry
        self.subentry = subentry
        # Mimic what LocalAiEntity.__init__ does
        self.model = subentry.data.get("model", subentry.data.get("CONF_MODEL"))


class TestLlamaCppUseLoadedModel:
    """Tests for the use_loaded_model async method in LlamaCppMixin."""

    async def test_disabled_uses_configured_model(self) -> None:
        """When use_loaded_model is False, return the configured model."""
        entry = MagicMock()
        subentry = _StubSubentry(
            {
                "model": "my-configured-model",
                CONF_LLAMACPP_CONFIG: {
                    CONF_LLAMACPP_USE_LOADED_MODEL: False,
                },
            }
        )

        class TestEntity(LlamaCppMixin, _StubBaseEntity):
            pass

        entity = TestEntity(entry, subentry)
        assert await entity._async_get_model(None) == "my-configured-model"

    async def test_enabled_queries_models_endpoint(self) -> None:
        """When use_loaded_model is True, query /models and return loaded model."""
        mock_loaded_model = MagicMock()
        mock_loaded_model.id = "qwen3.6-35b-a3b"
        mock_loaded_model.status.get.return_value = "loaded"

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [mock_loaded_model]
        mock_client.models.list = AsyncMock(return_value=mock_response)

        entry = _StubEntry(runtime_data=mock_client)
        subentry = _StubSubentry(
            {
                "model": "some-other-model",
                CONF_LLAMACPP_CONFIG: {
                    CONF_LLAMACPP_USE_LOADED_MODEL: True,
                },
            }
        )

        class TestEntity(LlamaCppMixin, _StubBaseEntity):
            pass

        entity = TestEntity(entry, subentry)
        assert await entity._async_get_model(None) == "qwen3.6-35b-a3b"

    async def test_prefers_config_model_if_loaded(self) -> None:
        """When config model is among loaded models, use config model."""
        mock_loaded_1 = MagicMock()
        mock_loaded_1.id = "other-model"
        mock_loaded_1.status.get.return_value = "loaded"

        mock_loaded_2 = MagicMock()
        mock_loaded_2.id = "my-configured-model"
        mock_loaded_2.status.get.return_value = "loaded"

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [mock_loaded_1, mock_loaded_2]
        mock_client.models.list = AsyncMock(return_value=mock_response)

        entry = _StubEntry(runtime_data=mock_client)
        subentry = _StubSubentry(
            {
                "model": "my-configured-model",
                CONF_LLAMACPP_CONFIG: {
                    CONF_LLAMACPP_USE_LOADED_MODEL: True,
                },
            }
        )

        class TestEntity(LlamaCppMixin, _StubBaseEntity):
            pass

        entity = TestEntity(entry, subentry)
        assert await entity._async_get_model(None) == "my-configured-model"

    async def test_falls_back_to_first_loaded_when_config_not_loaded(self) -> None:
        """When config model is not loaded, use first loaded model."""
        mock_loaded_1 = MagicMock()
        mock_loaded_1.id = "first-loaded-model"
        mock_loaded_1.status.get.return_value = "loaded"

        mock_loaded_2 = MagicMock()
        mock_loaded_2.id = "second-loaded-model"
        mock_loaded_2.status.get.return_value = "loaded"

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [mock_loaded_1, mock_loaded_2]
        mock_client.models.list = AsyncMock(return_value=mock_response)

        entry = _StubEntry(runtime_data=mock_client)
        subentry = _StubSubentry(
            {
                "model": "not-loaded-model",
                CONF_LLAMACPP_CONFIG: {
                    CONF_LLAMACPP_USE_LOADED_MODEL: True,
                },
            }
        )

        class TestEntity(LlamaCppMixin, _StubBaseEntity):
            pass

        entity = TestEntity(entry, subentry)
        assert await entity._async_get_model(None) == "first-loaded-model"

    async def test_fallback_on_api_failure(self) -> None:
        """When /models query fails, fall back to configured model."""
        mock_client = MagicMock()
        mock_client.models.list = AsyncMock(side_effect=Exception("Connection refused"))

        entry = _StubEntry(runtime_data=mock_client)
        subentry = _StubSubentry(
            {
                "model": "fallback-model",
                CONF_LLAMACPP_CONFIG: {
                    CONF_LLAMACPP_USE_LOADED_MODEL: True,
                },
            }
        )

        class TestEntity(LlamaCppMixin, _StubBaseEntity):
            pass

        entity = TestEntity(entry, subentry)
        assert await entity._async_get_model(None) == "fallback-model"

    async def test_no_loaded_models_fallback(self) -> None:
        """When no models are loaded, fall back to configured model."""
        mock_unloaded = MagicMock()
        mock_unloaded.id = "unloaded-model"
        mock_unloaded.status.value = "unloaded"

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [mock_unloaded]
        mock_client.models.list = AsyncMock(return_value=mock_response)

        entry = _StubEntry(runtime_data=mock_client)
        subentry = _StubSubentry(
            {
                "model": "fallback-model",
                CONF_LLAMACPP_CONFIG: {
                    CONF_LLAMACPP_USE_LOADED_MODEL: True,
                },
            }
        )

        class TestEntity(LlamaCppMixin, _StubBaseEntity):
            pass

        entity = TestEntity(entry, subentry)
        assert await entity._async_get_model(None) == "fallback-model"

    async def test_fresh_fetch_per_access(self) -> None:
        """Each access to _async_get_model triggers a fresh server query."""
        mock_loaded_model = MagicMock()
        mock_loaded_model.id = "fresh-model"
        mock_loaded_model.status.get.return_value = "loaded"

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [mock_loaded_model]
        mock_client.models.list = AsyncMock(return_value=mock_response)

        entry = _StubEntry(runtime_data=mock_client)
        subentry = _StubSubentry(
            {
                "model": "config-model",
                CONF_LLAMACPP_CONFIG: {
                    CONF_LLAMACPP_USE_LOADED_MODEL: True,
                },
            }
        )

        class TestEntity(LlamaCppMixin, _StubBaseEntity):
            pass

        entity = TestEntity(entry, subentry)
        first_call = await entity._async_get_model(None)
        mock_client.models.list.reset_mock()
        second_call = await entity._async_get_model(None)
        assert first_call == second_call == "fresh-model"
        mock_client.models.list.assert_called()

    async def test_missing_entry_runtime_data_uses_configured(self) -> None:
        """When entry has no runtime_data, fall back to configured model."""
        entry = _StubEntry(runtime_data=None)
        subentry = _StubSubentry(
            {
                "model": "config-model",
                CONF_LLAMACPP_CONFIG: {
                    CONF_LLAMACPP_USE_LOADED_MODEL: True,
                },
            }
        )

        class TestEntity(LlamaCppMixin, _StubBaseEntity):
            pass

        entity = TestEntity(entry, subentry)
        assert await entity._async_get_model(None) == "config-model"

    async def test_no_llamacpp_config_uses_configured(self) -> None:
        """When llamacpp_config is missing, fall back to configured model."""
        entry = MagicMock()
        subentry = _StubSubentry(
            {
                "model": "config-model",
            }
        )

        class TestEntity(LlamaCppMixin, _StubBaseEntity):
            pass

        entity = TestEntity(entry, subentry)
        assert await entity._async_get_model(None) == "config-model"

    async def test_modalities_empty_filters_nothing(self) -> None:
        """No image attachments returns first loaded model regardless of architecture."""
        mock_model = MagicMock()
        mock_model.id = "text-only-model"
        mock_model.status.get.return_value = "loaded"
        mock_model.architecture = {"input_modalities": ["text"]}

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [mock_model]
        mock_client.models.list = AsyncMock(return_value=mock_response)

        entry = _StubEntry(runtime_data=mock_client)
        subentry = _StubSubentry(
            {
                "model": "config-model",
                CONF_LLAMACPP_CONFIG: {
                    CONF_LLAMACPP_USE_LOADED_MODEL: True,
                },
            }
        )

        class TestEntity(LlamaCppMixin, _StubBaseEntity):
            pass

        entity = TestEntity(entry, subentry)
        assert await entity._async_get_model(_make_chat_log()) == "text-only-model"

    async def test_modalities_match_uses_configured(self) -> None:
        """Configured model is used when it is loaded and matches image modality."""
        mock_model = MagicMock()
        mock_model.id = "my-configured-model"
        mock_model.status.get.return_value = "loaded"
        mock_model.architecture = {"input_modalities": ["text", "image"]}

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [mock_model]
        mock_client.models.list = AsyncMock(return_value=mock_response)

        entry = _StubEntry(runtime_data=mock_client)
        subentry = _StubSubentry(
            {
                "model": "my-configured-model",
                CONF_LLAMACPP_CONFIG: {
                    CONF_LLAMACPP_USE_LOADED_MODEL: True,
                },
            }
        )

        class TestEntity(LlamaCppMixin, _StubBaseEntity):
            pass

        entity = TestEntity(entry, subentry)
        chat_log = _make_chat_log([["image/png"]])
        assert await entity._async_get_model(chat_log) == "my-configured-model"

    async def test_modalities_mismatch_uses_first_matching_loaded(self) -> None:
        """When config model doesn't match image modality, first matching loaded model is used."""
        mock_config_model = MagicMock()
        mock_config_model.id = "text-only-model"
        mock_config_model.status.get.return_value = "loaded"
        mock_config_model.architecture = {"input_modalities": ["text"]}

        mock_matching_model = MagicMock()
        mock_matching_model.id = "multimodal-model"
        mock_matching_model.status.get.return_value = "loaded"
        mock_matching_model.architecture = {"input_modalities": ["text", "image"]}

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [mock_config_model, mock_matching_model]
        mock_client.models.list = AsyncMock(return_value=mock_response)

        entry = _StubEntry(runtime_data=mock_client)
        subentry = _StubSubentry(
            {
                "model": "text-only-model",
                CONF_LLAMACPP_CONFIG: {
                    CONF_LLAMACPP_USE_LOADED_MODEL: True,
                },
            }
        )

        class TestEntity(LlamaCppMixin, _StubBaseEntity):
            pass

        entity = TestEntity(entry, subentry)
        chat_log = _make_chat_log([["image/png"]])
        assert await entity._async_get_model(chat_log) == "multimodal-model"

    async def test_modalities_no_match_falls_back_to_configured(self) -> None:
        """When no loaded model matches image modality, fall back to configured model."""
        mock_model = MagicMock()
        mock_model.id = "text-only-model"
        mock_model.status.get.return_value = "loaded"
        mock_model.architecture = {"input_modalities": ["text"]}

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [mock_model]
        mock_client.models.list = AsyncMock(return_value=mock_response)

        entry = _StubEntry(runtime_data=mock_client)
        subentry = _StubSubentry(
            {
                "model": "fallback-model",
                CONF_LLAMACPP_CONFIG: {
                    CONF_LLAMACPP_USE_LOADED_MODEL: True,
                },
            }
        )

        class TestEntity(LlamaCppMixin, _StubBaseEntity):
            pass

        entity = TestEntity(entry, subentry)
        chat_log = _make_chat_log([["image/png"]])
        assert await entity._async_get_model(chat_log) == "fallback-model"

    async def test_modalities_model_without_architecture_matches(self) -> None:
        """Models without architecture attribute are treated as matching all modalities."""
        mock_model = MagicMock()
        mock_model.id = "no-arch-model"
        mock_model.status.get.return_value = "loaded"
        del mock_model.architecture

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [mock_model]
        mock_client.models.list = AsyncMock(return_value=mock_response)

        entry = _StubEntry(runtime_data=mock_client)
        subentry = _StubSubentry(
            {
                "model": "no-arch-model",
                CONF_LLAMACPP_CONFIG: {
                    CONF_LLAMACPP_USE_LOADED_MODEL: True,
                },
            }
        )

        class TestEntity(LlamaCppMixin, _StubBaseEntity):
            pass

        entity = TestEntity(entry, subentry)
        chat_log = _make_chat_log([["image/png"]])
        assert await entity._async_get_model(chat_log) == "no-arch-model"
