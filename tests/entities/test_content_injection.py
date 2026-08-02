"""Unit tests for content injection methods in LocalAiEntity."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components import conversation as ha_conversation
from homeassistant.const import CONF_MODEL
from homeassistant.core import HomeAssistant

from custom_components.local_openai.const import (
    CONF_CONTENT_INJECTION_METHOD,
    CONF_CONTENT_INJECTION_METHOD_ASSISTANT,
    CONF_CONTENT_INJECTION_METHOD_TOOL,
    CONF_CONTENT_INJECTION_METHOD_USER,
    CONF_WEAVIATE_API_KEY,
    CONF_WEAVIATE_HOST,
    CONF_WEAVIATE_OPTIONS,
)
from custom_components.local_openai.conversation import LocalAiConversationEntity


def _make_messages(roles: list[str]) -> list[dict]:
    """Create a list of mock message dicts with given roles."""
    return [{"role": role} for role in roles]


def injection_config(method: str, extra: dict | None = None) -> dict:
    """Build subentry data dict for injection tests."""
    base = {CONF_MODEL: "test-model", CONF_CONTENT_INJECTION_METHOD: method}
    if extra:
        base.update(extra)
    return base


def weaviate_entry_opts(host: str, api_key: str | None = None) -> dict:
    """Build entry.data weaviate options dict."""
    opts = {CONF_WEAVIATE_HOST: host}
    if api_key:
        opts[CONF_WEAVIATE_API_KEY] = api_key
    return {CONF_WEAVIATE_OPTIONS: opts}


class MockSubentry:
    """Mock ConfigSubentry that allows updating data."""

    def __setattr__(self, name: str, value: object) -> None:
        object.__setattr__(self, name, value)

    def __init__(self, data: dict) -> None:
        self.subentry_id = "test_conversation_subentry_id"
        self.subentry_type = "conversation"
        self.title = "Conversation Agent"
        self.data = data
        self.unique_id = None


@pytest.fixture
def mock_entity(hass: HomeAssistant) -> LocalAiConversationEntity:
    """Create a mock conversation entity for testing content injection."""
    entry = MagicMock()
    entry.data = {}
    entry.runtime_data = MagicMock()

    subentry = MockSubentry({CONF_MODEL: "test-model"})

    entity = LocalAiConversationEntity(entry, subentry)
    entity.hass = hass
    return entity


@pytest.fixture
def mock_user_input() -> MagicMock:
    """Create a mock ConversationInput."""
    return MagicMock(spec=ha_conversation.ConversationInput)


class TestInjectContent:
    """Tests for the _inject_content static method."""

    def test_inject_content_tool_method(self):
        """Test injecting content as tool message."""
        messages = _make_messages(["system", "user", "assistant", "user"])
        inject_content = ["Some context info"]

        result = LocalAiConversationEntity._inject_content(
            CONF_CONTENT_INJECTION_METHOD_TOOL,
            inject_content,
            messages,
        )

        assert len(result) == 5
        injected = result[-2]
        assert injected["role"] == "tool"
        assert injected["tool_call_id"] == "injected_content"
        assert "Some context info" in injected["content"]
        assert "Contextual information" in injected["content"]
        assert result[-1]["role"] == "user"

    def test_inject_content_assistant_method(self):
        """Test injecting content as assistant message."""
        messages = _make_messages(["system", "user"])
        inject_content = ["Context line 1", "Context line 2"]

        result = LocalAiConversationEntity._inject_content(
            CONF_CONTENT_INJECTION_METHOD_ASSISTANT,
            inject_content,
            messages,
        )

        assert len(result) == 3
        injected = result[-2]
        assert injected["role"] == "assistant"
        assert "Context line 1" in injected["content"]
        assert "Context line 2" in injected["content"]
        assert "\n\n" in injected["content"]
        assert result[-1]["role"] == "user"

    def test_inject_content_user_method(self):
        """Test injecting content as user message."""
        messages = _make_messages(["system", "assistant", "user"])
        inject_content = ["User context"]

        result = LocalAiConversationEntity._inject_content(
            CONF_CONTENT_INJECTION_METHOD_USER,
            inject_content,
            messages,
        )

        assert len(result) == 4
        injected = result[-2]
        assert injected["role"] == "user"
        assert "User context" in injected["content"]
        assert result[-1]["role"] == "user"

    def test_inject_content_multiple_context_lines(self):
        """Test that multiple context lines are joined with newlines."""
        messages = _make_messages(["user"])
        inject_content = ["Line 1", "Line 2", "Line 3"]

        result = LocalAiConversationEntity._inject_content(
            CONF_CONTENT_INJECTION_METHOD_USER,
            inject_content,
            messages,
        )

        injected = result[-2]
        assert "Line 1" in injected["content"]
        assert "Line 2" in injected["content"]
        assert "Line 3" in injected["content"]
        assert "Line 1\n\nLine 2\n\nLine 3" in injected["content"]

    def test_inject_content_header_added(self):
        """Test that a header is prepended to inject_content."""
        messages = _make_messages(["user"])
        inject_content = ["Context"]

        result = LocalAiConversationEntity._inject_content(
            CONF_CONTENT_INJECTION_METHOD_USER,
            inject_content,
            messages,
        )

        injected = result[-2]
        assert "Contextual information to assist" in injected["content"]
        assert "Do not repeat or reference" in injected["content"]


class TestMaybeInjectContext:
    """Tests for the _maybe_inject_context async method."""

    def _make_tool(self, name: str) -> dict:
        """Create a mock tool dict."""
        return {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {"type": "object", "properties": {}},
            },
        }

    async def test_no_injection_when_method_none(self, mock_entity):
        """Test no injection when method is None."""
        messages = _make_messages(["system", "user"])
        tools = [self._make_tool("GetDateTime")]

        result_messages, result_tools = await mock_entity._maybe_inject_context(
            messages,
            None,
            tools,
            MagicMock(spec=ha_conversation.ConversationInput),
        )

        assert result_messages == messages
        assert result_tools == tools

    async def test_injection_injects_datetime(self, mock_entity):
        """Test that injection includes date and time."""
        mock_entity.subentry.data = injection_config(CONF_CONTENT_INJECTION_METHOD_USER)

        messages = _make_messages(["system", "user"])

        result_messages, _ = await mock_entity._maybe_inject_context(
            messages,
            CONF_CONTENT_INJECTION_METHOD_USER,
            [],
            MagicMock(spec=ha_conversation.ConversationInput),
        )

        assert len(result_messages) == 3
        injected = result_messages[-2]
        assert injected["role"] == "user"
        assert (
            "date" in injected["content"].lower()
            or "time" in injected["content"].lower()
        )

    async def test_injection_removes_getdatetime_tool(self, mock_entity):
        """Test that GetDateTime tool is removed when injection occurs."""
        mock_entity.subentry.data = injection_config(CONF_CONTENT_INJECTION_METHOD_USER)

        messages = _make_messages(["system", "user"])
        tools = [
            self._make_tool("GetDateTime"),
            self._make_tool("DoSomething"),
            self._make_tool("GetWeather"),
        ]

        result_messages, result_tools = await mock_entity._maybe_inject_context(
            messages,
            CONF_CONTENT_INJECTION_METHOD_USER,
            tools,
            MagicMock(spec=ha_conversation.ConversationInput),
        )

        tool_names = [t["function"]["name"] for t in result_tools]
        assert "GetDateTime" not in tool_names
        assert "DoSomething" in tool_names
        assert "GetWeather" in tool_names

    async def test_injection_preserves_other_tools(self, mock_entity):
        """Test that non-GetDateTime tools are preserved."""
        mock_entity.subentry.data = injection_config(
            CONF_CONTENT_INJECTION_METHOD_ASSISTANT
        )

        messages = _make_messages(["system", "user"])
        tools = [
            self._make_tool("GetDateTime"),
            self._make_tool("SearchFiles"),
            self._make_tool("SendEmail"),
        ]

        _, result_tools = await mock_entity._maybe_inject_context(
            messages,
            CONF_CONTENT_INJECTION_METHOD_ASSISTANT,
            tools,
            MagicMock(spec=ha_conversation.ConversationInput),
        )

        assert len(result_tools) == 2
        tool_names = [t["function"]["name"] for t in result_tools]
        assert "SearchFiles" in tool_names
        assert "SendEmail" in tool_names

    async def test_injection_tool_method(self, mock_entity):
        """Test injection with tool method."""
        mock_entity.subentry.data = injection_config(CONF_CONTENT_INJECTION_METHOD_TOOL)

        messages = _make_messages(["system", "user"])

        result_messages, _ = await mock_entity._maybe_inject_context(
            messages,
            CONF_CONTENT_INJECTION_METHOD_TOOL,
            [],
            MagicMock(spec=ha_conversation.ConversationInput),
        )

        injected = result_messages[-2]
        assert injected["role"] == "tool"

    async def test_injection_assistant_method(self, mock_entity):
        """Test injection with assistant method."""
        mock_entity.subentry.data = injection_config(
            CONF_CONTENT_INJECTION_METHOD_ASSISTANT
        )

        messages = _make_messages(["system", "user"])

        result_messages, _ = await mock_entity._maybe_inject_context(
            messages,
            CONF_CONTENT_INJECTION_METHOD_ASSISTANT,
            [],
            MagicMock(spec=ha_conversation.ConversationInput),
        )

        injected = result_messages[-2]
        assert injected["role"] == "assistant"

    async def test_injection_includes_weaviate_results(self, mock_entity):
        """Test that Weaviate RAG results are included in injected content."""
        mock_entity.subentry.data = injection_config(
            CONF_CONTENT_INJECTION_METHOD_USER,
            extra={
                CONF_WEAVIATE_OPTIONS: {
                    "weaviate_class_name": "TestClass",
                    "weaviate_max_results": "2",
                },
            },
        )
        mock_entity.entry.data = weaviate_entry_opts(
            "http://localhost:8080", "test-key"
        )

        messages = _make_messages(["system", "user"])

        mock_input = MagicMock(spec=ha_conversation.ConversationInput)
        mock_input.text = "test query"

        mock_client = MagicMock()
        mock_client.hybrid_search = AsyncMock(
            return_value=[
                {"query": "test query", "content": "relevant content here"},
            ]
        )

        with patch(
            "custom_components.local_openai.entity.WeaviateClient",
            return_value=mock_client,
        ):
            result_messages, _ = await mock_entity._maybe_inject_context(
                messages,
                CONF_CONTENT_INJECTION_METHOD_USER,
                [],
                mock_input,
            )

        injected = result_messages[-2]
        assert "relevant content here" in injected["content"]
        assert "test query" in injected["content"]

    async def test_injection_weaviate_error_handled(self, mock_entity):
        """Test that Weaviate errors are logged but injection still proceeds."""
        mock_entity.subentry.data = injection_config(
            CONF_CONTENT_INJECTION_METHOD_USER,
            extra={
                CONF_WEAVIATE_OPTIONS: {
                    "weaviate_class_name": "TestClass",
                },
            },
        )
        mock_entity.entry.data = {CONF_WEAVIATE_HOST: "http://localhost:8080"}

        messages = _make_messages(["system", "user"])

        mock_input = MagicMock(spec=ha_conversation.ConversationInput)
        mock_input.text = "test query"

        mock_client = MagicMock()
        mock_client.hybrid_search = AsyncMock(
            side_effect=Exception("Connection failed")
        )

        with patch(
            "custom_components.local_openai.entity.WeaviateClient",
            return_value=mock_client,
        ):
            result_messages, _ = await mock_entity._maybe_inject_context(
                messages,
                CONF_CONTENT_INJECTION_METHOD_USER,
                [],
                mock_input,
            )

        assert len(result_messages) == 3

    async def test_injection_no_weaviate_when_no_host(self, mock_entity):
        """Test that Weaviate is not queried when host is not configured."""
        mock_entity.subentry.data = injection_config(
            CONF_CONTENT_INJECTION_METHOD_USER,
            extra={
                CONF_WEAVIATE_OPTIONS: {
                    "weaviate_class_name": "TestClass",
                },
            },
        )
        mock_entity.entry.data = {}

        messages = _make_messages(["system", "user"])

        mock_input = MagicMock(spec=ha_conversation.ConversationInput)
        mock_input.text = "test query"

        with patch(
            "custom_components.local_openai.entity.WeaviateClient",
        ) as MockWeaviate:
            result_messages, _ = await mock_entity._maybe_inject_context(
                messages,
                CONF_CONTENT_INJECTION_METHOD_USER,
                [],
                mock_input,
            )

        assert MockWeaviate.call_count == 0
        assert len(result_messages) == 3
