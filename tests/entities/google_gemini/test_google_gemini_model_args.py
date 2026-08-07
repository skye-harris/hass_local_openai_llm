"""Unit tests for Google Gemini mixin initialization and model args."""

from __future__ import annotations

from custom_components.local_openai.entities.google_gemini import (
    GoogleGeminiMixin,
    get_ai_task_config_schema,
    get_conversation_config_schema,
)


class TestGoogleGeminiMixinInit:
    """Tests for GoogleGeminiMixin initialization."""

    def test_thought_signatures_starts_empty(self):
        """Test that _thought_signatures starts as empty dict."""
        mixin = GoogleGeminiMixin()
        assert mixin._thought_signatures == {}


class TestGoogleGeminiSchema:
    """Tests for Google Gemini schema functions."""

    def test_get_conversation_config_schema_returns_empty(self):
        """Test that conversation schema returns empty dict."""
        result = get_conversation_config_schema()
        assert result == {}

    def test_get_ai_task_config_schema_returns_empty(self):
        """Test that AI task schema returns empty dict."""
        result = get_ai_task_config_schema()
        assert result == {}
