"""Unit tests for Google Gemini entity registration and server type."""

from __future__ import annotations

from custom_components.local_openai.const import (
    SERVER_TYPE_GOOGLE_GEMINI,
    SERVER_TYPE_OPTIONS,
)
from custom_components.local_openai.entities.google_gemini import (
    GoogleGeminiAITaskEntity,
    GoogleGeminiConversationEntity,
)


class TestGoogleGeminiServerType:
    """Tests for Google Gemini server type constants."""

    def test_server_type_google_gemini_defined(self):
        """Test that SERVER_TYPE_GOOGLE_GEMINI constant exists."""
        assert SERVER_TYPE_GOOGLE_GEMINI == "google_gemini"

    def test_server_type_in_options(self):
        """Test that google_gemini is included in SERVER_TYPE_OPTIONS."""
        assert SERVER_TYPE_GOOGLE_GEMINI in SERVER_TYPE_OPTIONS

    def test_server_type_options_contains_expected_types(self):
        """Test that SERVER_TYPE_OPTIONS has expected server types."""
        expected = {
            "llama_cpp",
            "vllm",
            "deepseek",
            "localai",
            "google_gemini",
        }
        assert expected.issubset(set(SERVER_TYPE_OPTIONS))


class TestGoogleGeminiEntityClasses:
    """Tests for Google Gemini entity classes."""

    def test_conversation_entity_inherits_mixin(self):
        """Test that GoogleGeminiConversationEntity includes GoogleGeminiMixin."""
        from custom_components.local_openai.entities.google_gemini import (
            GoogleGeminiMixin,
        )

        assert GoogleGeminiMixin in GoogleGeminiConversationEntity.__mro__

    def test_ai_task_entity_inherits_mixin(self):
        """Test that GoogleGeminiAITaskEntity includes GoogleGeminiMixin."""
        from custom_components.local_openai.entities.google_gemini import (
            GoogleGeminiMixin,
        )

        assert GoogleGeminiMixin in GoogleGeminiAITaskEntity.__mro__

    def test_conversation_entity_has_mixin_methods(self):
        """Test that GoogleGeminiConversationEntity has mixin methods."""
        assert hasattr(GoogleGeminiConversationEntity, "_store_thought_signature")
        assert hasattr(GoogleGeminiConversationEntity, "_get_thought_signature")
        assert hasattr(GoogleGeminiConversationEntity, "_transform_stream")
        assert hasattr(
            GoogleGeminiConversationEntity, "_convert_content_to_chat_message"
        )

    def test_ai_task_entity_has_mixin_methods(self):
        """Test that GoogleGeminiAITaskEntity has mixin methods."""
        assert hasattr(GoogleGeminiAITaskEntity, "_store_thought_signature")
        assert hasattr(GoogleGeminiAITaskEntity, "_get_thought_signature")
        assert hasattr(GoogleGeminiAITaskEntity, "_transform_stream")
        assert hasattr(GoogleGeminiAITaskEntity, "_convert_content_to_chat_message")
