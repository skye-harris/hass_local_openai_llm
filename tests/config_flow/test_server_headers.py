"""Tests for config flow custom headers validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from custom_components.local_openai.config_flow import (
    LocalAiConfigFlow,
    _validate_server_headers,
)
from custom_components.local_openai.const import (
    CONF_CUSTOM_HEADERS,
    CONF_SERVER_HEADERS,
)
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType


@dataclass
class _ValidateHeaderTestCase:
    """Parametrized test case for _validate_server_headers."""

    name: str
    input_headers: list[dict[str, str]]
    expected: list[dict[str, str]]


_TEST_VALIDATE_CASES = [
    _ValidateHeaderTestCase(
        name="filters_empty_key",
        input_headers=[
            {"Key": "X-Valid", "Value": "abc"},
            {"Key": "", "Value": "should be removed"},
        ],
        expected=[{"Key": "X-Valid", "Value": "abc"}],
    ),
    _ValidateHeaderTestCase(
        name="filters_empty_value",
        input_headers=[
            {"Key": "X-Valid", "Value": "abc"},
            {"Key": "should be removed", "Value": ""},
        ],
        expected=[{"Key": "X-Valid", "Value": "abc"}],
    ),
    _ValidateHeaderTestCase(
        name="filters_whitespace_only_key",
        input_headers=[
            {"Key": "X-Valid", "Value": "abc"},
            {"Key": "   ", "Value": "should be removed"},
        ],
        expected=[{"Key": "X-Valid", "Value": "abc"}],
    ),
    _ValidateHeaderTestCase(
        name="filters_whitespace_only_value",
        input_headers=[
            {"Key": "X-Valid", "Value": "abc"},
            {"Key": "should be removed", "Value": "  "},
        ],
        expected=[{"Key": "X-Valid", "Value": "abc"}],
    ),
    _ValidateHeaderTestCase(
        name="filters_both_empty",
        input_headers=[{"Key": "", "Value": ""}],
        expected=[],
    ),
]


class TestValidateServerHeaders:
    """Tests for _validate_server_headers validator."""

    @pytest.mark.parametrize(
        ("case"),
        _TEST_VALIDATE_CASES,
        ids=lambda c: c.name,
    )
    def test_filters_invalid_entries(self, case: _ValidateHeaderTestCase) -> None:
        """Test that entries with empty/whitespace keys or values are filtered."""
        result = _validate_server_headers(case.input_headers)
        assert result == case.expected

    def test_preserves_valid_entries(self) -> None:
        """Test that valid header entries are preserved."""
        headers = [
            {"Key": "X-Custom", "Value": "abc"},
            {"Key": "Authorization", "Value": "Bearer token"},
        ]
        result = _validate_server_headers(headers)
        assert result == headers

    def test_empty_list(self) -> None:
        """Test that an empty list returns empty."""
        assert _validate_server_headers([]) == []

    def test_preserves_valid_trailing_whitespace(self) -> None:
        """Test that values with meaningful trailing whitespace are preserved."""
        headers = [{"Key": "X-Valid", "Value": "abc  "}]
        result = _validate_server_headers(headers)
        assert result == [{"Key": "X-Valid", "Value": "abc  "}]


@dataclass
class _HeaderTestCase:
    """Parametrized test case for config flow user step."""

    name: str
    headers_input: list[dict[str, str]] | None
    expected_headers: list[dict[str, str]] | None
    expect_model_list_call: bool = True


_TEST_CASES = [
    _HeaderTestCase(
        name="valid_headers",
        headers_input=[
            {"Key": "X-Custom", "Value": "abc"},
            {"Key": "Authorization", "Value": "Bearer token"},
        ],
        expected_headers=[
            {"Key": "X-Custom", "Value": "abc"},
            {"Key": "Authorization", "Value": "Bearer token"},
        ],
    ),
    _HeaderTestCase(
        name="empty_headers_list",
        headers_input=[],
        expected_headers=[],
    ),
    _HeaderTestCase(
        name="no_headers_section",
        headers_input=None,
        expected_headers=None,
        expect_model_list_call=False,
    ),
    _HeaderTestCase(
        name="mixed_valid_and_empty",
        headers_input=[
            {"Key": "X-Valid", "Value": "abc"},
            {"Key": "", "Value": "empty key"},
            {"Key": "empty value", "Value": ""},
        ],
        expected_headers=[{"Key": "X-Valid", "Value": "abc"}],
    ),
]


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Create a mock OpenAI client."""
    mock = MagicMock()
    mock.models.list = AsyncMock()
    return mock


async def test_config_flow_user_shows_form(hass: HomeAssistant) -> None:
    """Test that user step shows the form initially."""
    flow = LocalAiConfigFlow()
    flow.hass = hass
    flow._async_abort_entries_match = MagicMock()

    result = await flow.async_step_user()
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"


@pytest.mark.parametrize(
    ("case"),
    _TEST_CASES,
    ids=lambda c: c.name,
)
async def test_config_flow_user_with_headers(
    hass: HomeAssistant,
    mock_openai_client: MagicMock,
    case: _HeaderTestCase,
) -> None:
    """Test user step with various custom headers configurations."""
    user_input: dict[str, Any] = {
        "server_name": "Test Server",
        "base_url": "http://test:8080/v1",
        "api_key": "test-key",
    }
    if case.headers_input is not None:
        user_input[CONF_CUSTOM_HEADERS] = {
            CONF_SERVER_HEADERS: case.headers_input,
        }

    with (
        patch(
            "custom_components.local_openai.config_flow.AsyncOpenAI",
            return_value=mock_openai_client,
        ),
    ):
        flow = LocalAiConfigFlow()
        flow.hass = hass
        flow._async_abort_entries_match = MagicMock()

        result = await flow.async_step_user(user_input)

    assert result["type"] is FlowResultType.CREATE_ENTRY

    if case.expect_model_list_call:
        mock_openai_client.models.list.assert_called_once()

    if case.expected_headers is not None:
        assert (
            result["data"][CONF_CUSTOM_HEADERS][CONF_SERVER_HEADERS]
            == case.expected_headers
        )
