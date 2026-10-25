"""Tests for the Weaviate client."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.loader import async_get_integration

from custom_components.local_openai.const import DOMAIN
from custom_components.local_openai.weaviate import WeaviateClient

MANIFEST = Path(__file__).parents[1] / "custom_components" / DOMAIN / "manifest.json"


@pytest.mark.usefixtures("enable_custom_integrations")
async def test_headers_include_integration_version(hass: HomeAssistant) -> None:
    """The integration name and manifest version are sent to Weaviate."""
    await async_get_integration(hass, DOMAIN)
    version = json.loads(MANIFEST.read_text())["version"]

    headers = WeaviateClient(hass, "http://weaviate:8080", "key")._api_headers()

    assert (
        headers["X-Weaviate-Client-Integration"] == f"hass-local-openai-llm/{version}"
    )
    assert headers["Authorization"] == "Bearer key"


async def test_headers_fall_back_to_unknown_version(hass: HomeAssistant) -> None:
    """An unresolvable version must not break the client."""
    headers = WeaviateClient(hass, "http://weaviate:8080", None)._api_headers()

    assert headers["X-Weaviate-Client-Integration"] == "hass-local-openai-llm/unknown"
    assert "Authorization" not in headers
