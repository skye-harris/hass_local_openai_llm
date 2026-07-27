"""Fixtures for local_openai tests."""

import sys
from unittest.mock import MagicMock

# Mock turbojpeg BEFORE any HA imports
sys.modules["turbojpeg"] = MagicMock()
sys.modules["turbojpeg.TurboJPEG"] = MagicMock()

import pytest  # noqa: E402
from types import MappingProxyType  # noqa: E402

from homeassistant.config_entries import (  # noqa: E402
    ConfigEntry,
    ConfigEntryState,
    ConfigSubentry,
)
from homeassistant.components import conversation  # noqa: E402
from homeassistant.const import CONF_MODEL  # noqa: E402
from homeassistant.core import HomeAssistant  # noqa: E402

from custom_components.local_openai.const import (  # noqa: E402
    CONF_LLAMACPP_CONFIG,
    CONF_LLAMACPP_ENABLE_THINKING,
    SERVER_TYPE_LLAMACPP,
    CONF_LLAMACPP_ID_SLOT,
    CONF_LLAMACPP_INCLUDE_PRIOR_THINKING,
    CONF_LLAMACPP_MIN_P,
    CONF_LLAMACPP_PRESENCE_PENALTY,
    CONF_LLAMACPP_REPEAT_PENALTY,
    CONF_LLAMACPP_TOP_K,
    CONF_LLAMACPP_TOP_P,
)


@pytest.fixture
async def mock_config_entry(hass: HomeAssistant) -> ConfigEntry:
    """Create a mock config_entry."""
    entry = ConfigEntry(
        domain="local_openai",
        title="Test Server",
        data={CONF_MODEL: "test-model"},
        source="user",
        state=ConfigEntryState.LOADED,
        version=1,
        minor_version=1,
        discovery_keys=MappingProxyType({}),
        options=None,
        subentries_data=None,
        unique_id=None,
    )
    mock_client = MagicMock()
    entry.runtime_data = mock_client
    hass.config_entries._entries[entry.entry_id] = entry
    return entry


@pytest.fixture
async def mock_config_entry_llamacpp(hass: HomeAssistant) -> ConfigEntry:
    """Create a mock config_entry with llama.cpp server type."""
    entry = ConfigEntry(
        domain="local_openai",
        title="Test Llama.cpp Server",
        data={CONF_MODEL: "test-model", "server_type": SERVER_TYPE_LLAMACPP},
        source="user",
        state=ConfigEntryState.LOADED,
        version=1,
        minor_version=1,
        discovery_keys=MappingProxyType({}),
        options=None,
        subentries_data=None,
        unique_id=None,
    )
    mock_client = MagicMock()
    entry.runtime_data = mock_client
    hass.config_entries._entries[entry.entry_id] = entry
    return entry


@pytest.fixture
async def mock_subentry(mock_config_entry: ConfigEntry) -> ConfigSubentry:
    """Create a mock subentry with default configuration."""
    subentry = ConfigSubentry(
        subentry_id="test_subentry_id",
        subentry_type="ai_task_data",
        title="AI Task",
        data=MappingProxyType({CONF_MODEL: "test-model"}),
        unique_id=None,
    )
    # Store subentry in a mutable dict for the fixture
    mock_config_entry._subentries = {"test_subentry_id": subentry}
    return subentry


@pytest.fixture
async def mock_subentry_llamacpp(
    mock_config_entry_llamacpp: ConfigEntry,
) -> ConfigSubentry:
    """Create a mock subentry with llama.cpp configuration."""
    subentry = ConfigSubentry(
        subentry_id="test_subentry_id",
        subentry_type="ai_task_data",
        title="AI Task",
        data=MappingProxyType(
            {
                CONF_MODEL: "test-model",
                CONF_LLAMACPP_CONFIG: {
                    CONF_LLAMACPP_TOP_P: 0.9,
                    CONF_LLAMACPP_TOP_K: 50,
                    CONF_LLAMACPP_MIN_P: 0.05,
                    CONF_LLAMACPP_REPEAT_PENALTY: 1.1,
                    CONF_LLAMACPP_PRESENCE_PENALTY: 0.5,
                    CONF_LLAMACPP_ID_SLOT: 1,
                    CONF_LLAMACPP_ENABLE_THINKING: True,
                    CONF_LLAMACPP_INCLUDE_PRIOR_THINKING: True,
                },
            }
        ),
        unique_id=None,
    )
    mock_config_entry_llamacpp._subentries = {"test_subentry_id": subentry}
    return subentry


@pytest.fixture
async def mock_conversation_subentry(
    mock_config_entry: ConfigEntry,
) -> ConfigSubentry:
    """Create a mock subentry for conversation agent."""
    subentry = ConfigSubentry(
        subentry_id="test_conversation_subentry_id",
        subentry_type="conversation",
        title="Conversation Agent",
        data=MappingProxyType({CONF_MODEL: "test-model"}),
        unique_id=None,
    )
    mock_config_entry._subentries = {
        "test_conversation_subentry_id": subentry,
    }
    return subentry


@pytest.fixture
async def mock_conversation_subentry_llamacpp(
    mock_config_entry_llamacpp: ConfigEntry,
) -> ConfigSubentry:
    """Create a mock subentry for conversation agent with llama.cpp config."""

    subentry = ConfigSubentry(
        subentry_id="test_conversation_subentry_id",
        subentry_type="conversation",
        title="Conversation Agent",
        data=MappingProxyType(
            {
                CONF_MODEL: "test-model",
                CONF_LLAMACPP_CONFIG: {
                    CONF_LLAMACPP_ENABLE_THINKING: False,
                    CONF_LLAMACPP_INCLUDE_PRIOR_THINKING: False,
                },
            }
        ),
        unique_id=None,
    )
    mock_config_entry_llamacpp._subentries = {
        "test_conversation_subentry_id": subentry,
    }
    return subentry


@pytest.fixture
def mock_llm_api():
    """Create a mock LLM API object."""
    api = MagicMock()
    api.id = "test_llm_api"
    return api


class MockAssistantContent(conversation.AssistantContent):
    """Mock AssistantContent that allows setting thinking_content."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__setattr__ = lambda self, name, value: setattr(self, name, value)

    def __init__(
        self,
        agent_id: str = "test_agent",
        content: str | None = "test",
        thinking_content: str | None = None,
        tool_calls: list | None = None,
    ):
        super().__init__(
            agent_id=agent_id,
            content=content,
            thinking_content=thinking_content,
            tool_calls=tool_calls,
        )


class MockConfigSubentry(ConfigSubentry):
    """Mock ConfigSubentry that allows updating data."""

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

    def __init__(self, data: dict):
        self.subentry_id = "test_conversation_subentry_id"
        self.subentry_type = "conversation"
        self.title = "Conversation Agent"
        self.data = data
        self.unique_id = None
