"""Tests for the decision model AI Task entity."""

from __future__ import annotations

import base64
import json
from types import MappingProxyType
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import httpx
import pytest
from homeassistant.components import ai_task
from homeassistant.components.conversation import Attachment
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import CONF_MODEL
from homeassistant.exceptions import HomeAssistantError
from openai import AsyncOpenAI

from custom_components.local_openai.ai_task import (
    LocalAIEvaluationEntity,
    LocalAITaskEntity,
    async_setup_entry,
)
from custom_components.local_openai.const import (
    CONF_CUSTOM_HEADERS,
    CONF_SERVER_HEADERS,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from homeassistant.core import HomeAssistant

QUESTIONS = {
    "delivered": {"type": "noul", "instructions": "Has the package arrived?"},
}
ANSWERS = {"answers": {"delivered": {"type": "noul", "noul": 0.9}}}


def _make_entry(
    data: dict | None = None,
    subentries: list[ConfigSubentry] | None = None,
) -> ConfigEntry:
    """Create a minimal config entry."""
    return ConfigEntry(
        domain="local_openai",
        title="Test Server",
        data=data or {},
        source="user",
        version=4,
        minor_version=0,
        discovery_keys=MappingProxyType({}),
        options=None,
        subentries_data=[subentry.as_dict() for subentry in subentries or []],
        unique_id=None,
    )


def _make_subentry(subentry_type: str, subentry_id: str) -> ConfigSubentry:
    """Create an AI Task subentry."""
    return ConfigSubentry(
        subentry_id=subentry_id,
        subentry_type=subentry_type,
        title="AI Task",
        data=MappingProxyType({CONF_MODEL: "openjev"}),
        unique_id=None,
    )


def _make_entity(
    hass: HomeAssistant,
    handler: Callable[[httpx.Request], httpx.Response],
    entry_data: dict | None = None,
) -> LocalAIEvaluationEntity:
    """Create an evaluation entity whose API requests are answered by handler."""
    entry = _make_entry(entry_data)
    entry.runtime_data = AsyncOpenAI(
        base_url="http://localhost:8080/v1",
        api_key="local-openai",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    entity = LocalAIEvaluationEntity(
        entry,
        _make_subentry("ai_task_evaluate", "test_evaluate_subentry_id"),
    )
    entity.hass = hass
    return entity


def _task(attachments: list[Attachment] | None = None) -> ai_task.EvaluationTask:
    """Create an evaluation task."""
    return ai_task.EvaluationTask(
        name="Check delivery",
        questions=QUESTIONS,
        state="Your package was left at the front door.",
        attachments=attachments,
    )


def test_supported_features(hass: HomeAssistant) -> None:
    """Test the entity only supports evaluation, with attachments."""
    entity = _make_entity(hass, lambda _: httpx.Response(200, json=ANSWERS))
    assert entity.supported_features == (
        ai_task.AITaskEntityFeature.EVALUATE
        | ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS
    )


async def test_evaluate_posts_to_systemone(hass: HomeAssistant) -> None:
    """Test the task is sent to /v1/systemone and the answers are returned."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=ANSWERS)

    result = await _make_entity(hass, handler)._async_evaluate(_task())

    assert result.answers == {"delivered": ai_task.NoulAnswer(noul=0.9)}
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert str(requests[0].url) == "http://localhost:8080/v1/systemone"
    assert json.loads(requests[0].content) == {
        "model": "openjev",
        "state": "Your package was left at the front door.",
        "questions": QUESTIONS,
    }


async def test_evaluate_sends_custom_headers(hass: HomeAssistant) -> None:
    """Test server custom headers are sent with evaluation requests."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=ANSWERS)

    entity = _make_entity(
        hass,
        handler,
        {
            CONF_CUSTOM_HEADERS: {
                CONF_SERVER_HEADERS: [{"Key": "X-Test", "Value": "abc"}]
            }
        },
    )
    await entity._async_evaluate(_task())

    assert requests[0].headers["X-Test"] == "abc"


async def test_evaluate_sends_image_attachments(
    hass: HomeAssistant,
    tmp_path: Path,
) -> None:
    """Test image attachments are sent as data URLs."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=ANSWERS)

    image = tmp_path / "snapshot.png"
    image.write_bytes(b"\x89PNG")
    attachment = Attachment(
        media_content_id="media-source://camera/camera.front_door",
        mime_type="image/png",
        path=image,
    )
    await _make_entity(hass, handler)._async_evaluate(_task([attachment]))

    encoded = base64.b64encode(b"\x89PNG").decode()
    assert json.loads(requests[0].content)["images"] == [
        f"data:image/png;base64,{encoded}",
    ]


async def test_not_a_decision_model_raises(hass: HomeAssistant) -> None:
    """Test the server rejecting the model as a decision model raises a clear error."""
    entity = _make_entity(
        hass,
        lambda _: httpx.Response(501, json={"error": {"message": "not supported"}}),
    )
    with pytest.raises(HomeAssistantError, match="openjev is not a decision model"):
        await entity._async_evaluate(_task())


async def test_api_error_raises(hass: HomeAssistant) -> None:
    """Test other API errors raise."""
    entity = _make_entity(
        hass,
        lambda _: httpx.Response(500, json={"error": {"message": "boom"}}),
    )
    with pytest.raises(HomeAssistantError, match="Error talking to API"):
        await entity._async_evaluate(_task())


async def test_setup_entry_creates_entity_per_subentry_type(
    hass: HomeAssistant,
) -> None:
    """Test each AI Task subentry type gets its own entity class."""
    entry = _make_entry(
        subentries=[
            _make_subentry("ai_task_data", "data"),
            _make_subentry("ai_task_evaluate", "evaluate"),
            _make_subentry("conversation", "conversation"),
        ],
    )
    entry.runtime_data = MagicMock()
    async_add_entities = MagicMock()

    await async_setup_entry(hass, entry, async_add_entities)

    added = {
        call.kwargs["config_subentry_id"]: call.args[0]
        for call in async_add_entities.call_args_list
    }
    assert set(added) == {"data", "evaluate"}
    assert isinstance(added["data"][0], LocalAITaskEntity)
    assert isinstance(added["evaluate"][0], LocalAIEvaluationEntity)
