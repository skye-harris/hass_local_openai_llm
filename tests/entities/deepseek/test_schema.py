"""Unit tests for the DeepSeek schema functions."""

from __future__ import annotations

import pytest
import voluptuous as vol
from custom_components.local_openai.const import (
    CONF_DEEPSEEK_REASONING_EFFORT,
    CONF_DEEPSEEK_REASONING_EFFORT_DEFAULT,
    CONF_DEEPSEEK_THINKING_MODE,
    CONF_DEEPSEEK_THINKING_MODE_ALWAYS,
    CONF_DEEPSEEK_THINKING_MODE_DISABLED,
    CONF_DEEPSEEK_THINKING_MODE_FIRST_LOOP,
)
from custom_components.local_openai.entities.deepseek import _get_deepseek_schema


def test_get_deepseek_schema_returns_expected_fields() -> None:
    """Test that schema includes expected DeepSeek fields."""
    schema = _get_deepseek_schema()
    assert CONF_DEEPSEEK_THINKING_MODE in schema
    assert CONF_DEEPSEEK_REASONING_EFFORT in schema


def test_reasoning_effort_visible_condition() -> None:
    """Test reasoning effort is only shown while a thinking mode enabling thinking is set."""
    schema = _get_deepseek_schema()
    marker = next(key for key in schema if key == CONF_DEEPSEEK_REASONING_EFFORT)
    assert marker.visible == {
        "field": CONF_DEEPSEEK_THINKING_MODE,
        "operator": "in",
        "value": [
            CONF_DEEPSEEK_THINKING_MODE_ALWAYS,
            CONF_DEEPSEEK_THINKING_MODE_FIRST_LOOP,
        ],
    }


def _validator() -> vol.Schema:
    return vol.Schema(_get_deepseek_schema())


class TestValidation:
    """Validation tests for _get_deepseek_schema."""

    def test_reasoning_effort_defaults_when_missing(self) -> None:
        validator = _validator()
        result = validator({})
        assert result[CONF_DEEPSEEK_REASONING_EFFORT] == (
            CONF_DEEPSEEK_REASONING_EFFORT_DEFAULT
        )

    def test_thinking_mode_accepts_valid_values(self) -> None:
        validator = _validator()
        for value in (
            CONF_DEEPSEEK_THINKING_MODE_ALWAYS,
            CONF_DEEPSEEK_THINKING_MODE_FIRST_LOOP,
            CONF_DEEPSEEK_THINKING_MODE_DISABLED,
        ):
            result = validator({CONF_DEEPSEEK_THINKING_MODE: value})
            assert result[CONF_DEEPSEEK_THINKING_MODE] == value

    def test_thinking_mode_rejects_invalid_value(self) -> None:
        validator = _validator()
        with pytest.raises(vol.Invalid):
            validator({CONF_DEEPSEEK_THINKING_MODE: "invalid"})
