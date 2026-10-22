"""Tests for custom headers in async_setup_entry."""

from types import MappingProxyType
from unittest.mock import AsyncMock, MagicMock, patch

from custom_components.local_openai import async_migrate_entry, async_setup_entry
from custom_components.local_openai.const import (
    CONF_BASE_URL,
    CONF_CUSTOM_HEADERS,
    CONF_MAX_MESSAGE_HISTORY,
    CONF_SERVER_HEADERS,
)
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
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


def _make_entry(data: dict, version: int = 1) -> ConfigEntry:
    """Create a minimal ConfigEntry with the given data."""
    entry = ConfigEntry(
        domain="local_openai",
        title="Test",
        data=data,
        source="user",
        version=version,
        minor_version=0,
        discovery_keys=MappingProxyType({}),
        options=None,
        subentries_data=None,
        unique_id=None,
    )
    return entry


def _make_conversation_subentry(
    data: dict,
    subentry_id: str = "test_conversation_subentry_id",
) -> ConfigSubentry:
    """Create a mock conversation subentry."""
    return ConfigSubentry(
        subentry_id=subentry_id,
        subentry_type="conversation",
        title="Conversation Agent",
        data=MappingProxyType(data),
        unique_id=None,
    )


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


def _set_subentries(entry: ConfigEntry, subentries: dict) -> None:
    """Properly set subentries on a ConfigEntry using object.__setattr__."""
    object.__setattr__(entry, "subentries", MappingProxyType(subentries))


async def test_migrate_v2_to_v3_max_message_history_zero(
    hass: HomeAssistant,
) -> None:
    """Test migration of max_message_history=0 → unset (None)."""
    entry = _make_entry(
        {CONF_MODEL: "test-model", CONF_BASE_URL: "http://test:8080/v1"},
        version=2,
    )
    subentry = _make_conversation_subentry(
        {CONF_MODEL: "test-model", CONF_MAX_MESSAGE_HISTORY: 0},
    )
    _set_subentries(entry, {"test_conversation_subentry_id": subentry})
    hass.config_entries._entries[entry.entry_id] = entry

    result = await async_migrate_entry(hass, entry)

    assert result is True
    assert entry.version == 3
    updated_subentry = entry.subentries["test_conversation_subentry_id"]
    assert CONF_MAX_MESSAGE_HISTORY not in updated_subentry.data


async def test_migrate_v2_to_v3_max_message_history_negative(
    hass: HomeAssistant,
) -> None:
    """Test migration of max_message_history=-1 → unset (None)."""
    entry = _make_entry(
        {CONF_MODEL: "test-model", CONF_BASE_URL: "http://test:8080/v1"},
        version=2,
    )
    subentry = _make_conversation_subentry(
        {CONF_MODEL: "test-model", CONF_MAX_MESSAGE_HISTORY: -1},
    )
    _set_subentries(entry, {"test_conversation_subentry_id": subentry})
    hass.config_entries._entries[entry.entry_id] = entry

    result = await async_migrate_entry(hass, entry)

    assert result is True
    assert entry.version == 3
    updated_subentry = entry.subentries["test_conversation_subentry_id"]
    assert CONF_MAX_MESSAGE_HISTORY not in updated_subentry.data


async def test_migrate_v2_to_v3_max_message_history_string_zero(
    hass: HomeAssistant,
) -> None:
    """Test migration of max_message_history='0' → unset (None)."""
    entry = _make_entry(
        {CONF_MODEL: "test-model", CONF_BASE_URL: "http://test:8080/v1"},
        version=2,
    )
    subentry = _make_conversation_subentry(
        {CONF_MODEL: "test-model", CONF_MAX_MESSAGE_HISTORY: "0"},
    )
    _set_subentries(entry, {"test_conversation_subentry_id": subentry})
    hass.config_entries._entries[entry.entry_id] = entry

    result = await async_migrate_entry(hass, entry)

    assert result is True
    assert entry.version == 3
    updated_subentry = entry.subentries["test_conversation_subentry_id"]
    assert CONF_MAX_MESSAGE_HISTORY not in updated_subentry.data


async def test_migrate_v2_to_v3_max_message_history_not_conversation(
    hass: HomeAssistant,
) -> None:
    """Test that non-conversation subentries are not migrated."""
    entry = _make_entry(
        {CONF_MODEL: "test-model", CONF_BASE_URL: "http://test:8080/v1"},
        version=2,
    )
    subentry = ConfigSubentry(
        subentry_id="test_ai_task_subentry_id",
        subentry_type="ai_task",
        title="AI Task",
        data=MappingProxyType({CONF_MODEL: "test-model"}),
        unique_id=None,
    )
    _set_subentries(entry, {"test_ai_task_subentry_id": subentry})
    hass.config_entries._entries[entry.entry_id] = entry

    result = await async_migrate_entry(hass, entry)

    assert result is True
    assert entry.version == 3
    ai_task_subentry = entry.subentries["test_ai_task_subentry_id"]
    assert ai_task_subentry.data.get(CONF_MODEL) == "test-model"


async def test_migrate_v2_to_v3_no_migration_needed(
    hass: HomeAssistant,
) -> None:
    """Test that entries without max_message_history=0/-1 still bump to v3."""
    entry = _make_entry(
        {CONF_MODEL: "test-model", CONF_BASE_URL: "http://test:8080/v1"},
        version=2,
    )
    subentry = _make_conversation_subentry(
        {CONF_MODEL: "test-model", CONF_MAX_MESSAGE_HISTORY: 5},
    )
    _set_subentries(entry, {"test_conversation_subentry_id": subentry})
    hass.config_entries._entries[entry.entry_id] = entry

    result = await async_migrate_entry(hass, entry)

    assert result is True
    assert entry.version == 3
    updated_subentry = entry.subentries["test_conversation_subentry_id"]
    assert updated_subentry.data.get(CONF_MAX_MESSAGE_HISTORY) == 5


async def test_migrate_v2_to_v3_no_subentries(
    hass: HomeAssistant,
) -> None:
    """Test migration with no subentries."""
    entry = _make_entry(
        {CONF_MODEL: "test-model", CONF_BASE_URL: "http://test:8080/v1"},
        version=2,
    )
    hass.config_entries._entries[entry.entry_id] = entry

    result = await async_migrate_entry(hass, entry)

    assert result is True
    assert entry.version == 3
