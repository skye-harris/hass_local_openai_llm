"""Tests for the Weaviate client."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.loader import async_get_integration

from custom_components.local_openai.const import DOMAIN
from custom_components.local_openai.weaviate import WeaviateClient

MANIFEST = Path(__file__).parents[1] / "custom_components" / DOMAIN / "manifest.json"


class _FakeWeaviateResponse:
    """Minimal aiohttp response stand-in for a Weaviate GraphQL call."""

    status = 200

    async def __aenter__(self) -> _FakeWeaviateResponse:
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False

    def raise_for_status(self) -> None:
        """Do nothing, the fake response is always successful."""

    async def json(self) -> dict[str, Any]:
        """Return an empty result set for the queried class."""
        return {"data": {"Get": {"Homeassistant": []}}}


class _FakeWeaviateSession:
    """Record the GraphQL payloads the client sends."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def post(self, *args: object, **kwargs: object) -> _FakeWeaviateResponse:
        """Record the request body and return an empty result set."""
        payload = kwargs["json"]
        assert isinstance(payload, dict)
        self.payloads.append(payload)
        return _FakeWeaviateResponse()


def _client(
    hass: HomeAssistant,
) -> tuple[WeaviateClient, _FakeWeaviateSession]:
    """Return a client wired to a recording fake session."""
    client = WeaviateClient(hass, "http://weaviate:8080", None)
    session = _FakeWeaviateSession()
    client._aiohttp = session  # noqa: SLF001
    return client, session


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


@pytest.mark.parametrize(
    "query",
    [
        'Where is the "cat food" kept?',
        "Question on one line\nand a second line",
        r"C:\temp\new folder",
        'Unbalanced " quote',
    ],
    ids=["double-quote", "newline", "backslash", "unbalanced-quote"],
)
async def test_hybrid_search_escapes_query_text(
    hass: HomeAssistant,
    query: str,
) -> None:
    """User text is embedded as a valid GraphQL string literal."""
    client, session = _client(hass)

    await client.hybrid_search("Homeassistant", query, 0.5, 0.0, 2)

    graphql = session.payloads[0]["query"]
    assert f"query: {json.dumps(query)}," in graphql
    assert f'query: "{query}"' not in graphql


async def test_near_text_escapes_query_text(hass: HomeAssistant) -> None:
    """User text is embedded as a valid GraphQL string literal."""
    client, session = _client(hass)
    query = 'Where is the "cat food" kept?'

    await client.near_text("Homeassistant", query, 0.9, 2)

    graphql = session.payloads[0]["query"]
    assert f"concepts: [{json.dumps(query)}]" in graphql
    assert f'concepts: ["{query}"]' not in graphql
