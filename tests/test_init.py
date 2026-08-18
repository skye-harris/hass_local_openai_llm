"""Tests for custom headers in async_setup_entry."""

from unittest.mock import AsyncMock, MagicMock, patch

from custom_components.local_openai import async_setup_entry
from custom_components.local_openai.const import (
    CONF_BASE_URL,
    CONF_CUSTOM_HEADERS,
    CONF_SERVER_HEADERS,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_MODEL
from homeassistant.core import HomeAssistant
from homeassistant.helpers.httpx_client import get_async_client


def _make_mock_client() -> MagicMock:
    """Create a properly configured mock AsyncOpenAI client."""
    mock_client_instance = MagicMock()
    mock_client_instance.with_options.return_value.models.list = MagicMock(
        __aiter__=MagicMock(
            return_value=AsyncMock(__anext__=AsyncMock(side_effect=StopAsyncIteration))
        )
    )
    return mock_client_instance


def _make_entry(data: dict) -> ConfigEntry:
    """Create a minimal ConfigEntry with the given data."""
    from types import MappingProxyType

    entry = ConfigEntry(
        domain="local_openai",
        title="Test",
        data=data,
        source="user",
        version=1,
        minor_version=1,
        discovery_keys=MappingProxyType({}),
        options=None,
        subentries_data=None,
        unique_id=None,
    )
    return entry


async def test_setup_entry_without_custom_headers(hass: HomeAssistant) -> None:
    """Test setup entry passes extra_headers=None when no custom headers configured."""
    from openai import AsyncOpenAI

    entry = _make_entry(
        {CONF_MODEL: "test-model", CONF_BASE_URL: "http://test:8080/v1"}
    )

    with (
        patch.object(
            AsyncOpenAI, "__new__", return_value=_make_mock_client()
        ) as mock_openai,
        patch.object(
            hass.config_entries, "async_forward_entry_setups", return_value=True
        ),
    ):
        result = await async_setup_entry(hass, entry)

    assert result is True
    mock_openai.assert_called_once()
    call_kwargs = mock_openai.call_args
    assert call_kwargs.kwargs.get("default_headers") is None


async def test_setup_entry_with_custom_headers(hass: HomeAssistant) -> None:
    """Test setup entry passes extra_headers dict when custom headers are configured."""
    from openai import AsyncOpenAI

    entry = _make_entry(
        {
            CONF_MODEL: "test-model",
            CONF_BASE_URL: "http://test:8080/v1",
            CONF_CUSTOM_HEADERS: {
                CONF_SERVER_HEADERS: [
                    {"Key": "X-Custom-Header", "Value": "custom-value"},
                    {"Key": "X-Another", "Value": "another-value"},
                ]
            },
        }
    )

    with (
        patch.object(
            AsyncOpenAI, "__new__", return_value=_make_mock_client()
        ) as mock_openai,
        patch.object(
            hass.config_entries, "async_forward_entry_setups", return_value=True
        ),
    ):
        result = await async_setup_entry(hass, entry)

    assert result is True
    call_kwargs = mock_openai.call_args
    assert call_kwargs.kwargs["default_headers"] == {
        "X-Custom-Header": "custom-value",
        "X-Another": "another-value",
    }


async def test_setup_entry_filters_empty_keys(hass: HomeAssistant) -> None:
    """Test that valid headers are passed through when configured."""
    from openai import AsyncOpenAI

    entry = _make_entry(
        {
            CONF_MODEL: "test-model",
            CONF_BASE_URL: "http://test:8080/v1",
            CONF_CUSTOM_HEADERS: {
                CONF_SERVER_HEADERS: [
                    {"Key": "X-Valid", "Value": "valid-value"},
                ]
            },
        }
    )

    with (
        patch.object(
            AsyncOpenAI, "__new__", return_value=_make_mock_client()
        ) as mock_openai,
        patch.object(
            hass.config_entries, "async_forward_entry_setups", return_value=True
        ),
    ):
        result = await async_setup_entry(hass, entry)

    assert result is True
    call_kwargs = mock_openai.call_args
    assert call_kwargs.kwargs["default_headers"] == {"X-Valid": "valid-value"}


async def test_setup_entry_empty_headers_list(hass: HomeAssistant) -> None:
    """Test that an empty custom_headers list results in extra_headers=None."""
    from openai import AsyncOpenAI

    entry = _make_entry(
        {
            CONF_MODEL: "test-model",
            CONF_BASE_URL: "http://test:8080/v1",
            CONF_CUSTOM_HEADERS: {
                CONF_SERVER_HEADERS: [],
            },
        }
    )

    with (
        patch.object(
            AsyncOpenAI, "__new__", return_value=_make_mock_client()
        ) as mock_openai,
        patch.object(
            hass.config_entries, "async_forward_entry_setups", return_value=True
        ),
    ):
        result = await async_setup_entry(hass, entry)

    assert result is True
    call_kwargs = mock_openai.call_args
    assert call_kwargs.kwargs.get("default_headers") is None


async def test_setup_entry_http_client_passed(hass: HomeAssistant) -> None:
    """Test that the hass httpx client is passed to AsyncOpenAI."""
    from openai import AsyncOpenAI

    entry = _make_entry(
        {CONF_MODEL: "test-model", CONF_BASE_URL: "http://test:8080/v1"}
    )

    with (
        patch.object(
            AsyncOpenAI, "__new__", return_value=_make_mock_client()
        ) as mock_openai,
        patch.object(
            hass.config_entries, "async_forward_entry_setups", return_value=True
        ),
    ):
        await async_setup_entry(hass, entry)

    call_kwargs = mock_openai.call_args
    assert call_kwargs.kwargs["http_client"] is get_async_client(hass)


async def test_setup_entry_duplicate_keys(hass: HomeAssistant) -> None:
    """Test that duplicate header keys result in last-one-wins behavior."""
    from openai import AsyncOpenAI

    entry = _make_entry(
        {
            CONF_MODEL: "test-model",
            CONF_BASE_URL: "http://test:8080/v1",
            CONF_CUSTOM_HEADERS: {
                CONF_SERVER_HEADERS: [
                    {"Key": "X-Custom", "Value": "first-value"},
                    {"Key": "X-Custom", "Value": "second-value"},
                ]
            },
        }
    )

    with (
        patch.object(
            AsyncOpenAI, "__new__", return_value=_make_mock_client()
        ) as mock_openai,
        patch.object(
            hass.config_entries, "async_forward_entry_setups", return_value=True
        ),
    ):
        result = await async_setup_entry(hass, entry)

    assert result is True
    call_kwargs = mock_openai.call_args
    assert call_kwargs.kwargs["default_headers"] == {"X-Custom": "second-value"}


async def test_setup_entry_empty_values(hass: HomeAssistant) -> None:
    """Test that headers with non-empty values are passed through."""
    from openai import AsyncOpenAI

    entry = _make_entry(
        {
            CONF_MODEL: "test-model",
            CONF_BASE_URL: "http://test:8080/v1",
            CONF_CUSTOM_HEADERS: {
                CONF_SERVER_HEADERS: [
                    {"Key": "X-NonEmpty", "Value": "has-value"},
                ]
            },
        }
    )

    with (
        patch.object(
            AsyncOpenAI, "__new__", return_value=_make_mock_client()
        ) as mock_openai,
        patch.object(
            hass.config_entries, "async_forward_entry_setups", return_value=True
        ),
    ):
        result = await async_setup_entry(hass, entry)

    assert result is True
    call_kwargs = mock_openai.call_args
    assert call_kwargs.kwargs["default_headers"] == {
        "X-NonEmpty": "has-value",
    }
