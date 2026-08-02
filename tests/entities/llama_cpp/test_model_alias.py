"""Unit tests for get_model_alias function."""

from __future__ import annotations

from unittest.mock import MagicMock

from custom_components.local_openai.entities.llama_cpp import get_model_alias


def test_get_model_alias_returns_alias_when_set():
    """Test that alias is returned when set."""
    model = MagicMock(alias="my-alias")
    assert get_model_alias(model) == "my-alias"


def test_get_model_alias_returns_none_when_alias_is_none():
    """Test that None is returned when alias is not set."""
    model = type("MockModel", (), {"alias": None})()
    assert get_model_alias(model) is None
