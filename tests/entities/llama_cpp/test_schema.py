"""Unit tests for llama.cpp entity mixin and schema functions."""

from __future__ import annotations

import pytest
import voluptuous as vol
from custom_components.local_openai.const import (
    CONF_LLAMACPP_ID_SLOT,
    CONF_LLAMACPP_INCLUDE_PRIOR_THINKING,
    CONF_LLAMACPP_MIN_P,
    CONF_LLAMACPP_PRESENCE_PENALTY,
    CONF_LLAMACPP_REPEAT_PENALTY,
    CONF_LLAMACPP_THINKING_MODE,
    CONF_LLAMACPP_THINKING_MODE_ALWAYS,
    CONF_LLAMACPP_THINKING_MODE_FIRST_LOOP,
    CONF_LLAMACPP_TOP_K,
    CONF_LLAMACPP_TOP_P,
    CONF_LLAMACPP_USE_LOADED_MODEL,
)
from custom_components.local_openai.entities.llama_cpp import (
    _get_llama_cpp_schema,
)


def test_get_llama_cpp_schema_returns_expected_fields():
    """Test that schema includes all expected llama.cpp fields."""
    schema = _get_llama_cpp_schema()
    expected_fields = [
        CONF_LLAMACPP_THINKING_MODE,
        CONF_LLAMACPP_INCLUDE_PRIOR_THINKING,
        CONF_LLAMACPP_ID_SLOT,
        CONF_LLAMACPP_TOP_P,
        CONF_LLAMACPP_TOP_K,
        CONF_LLAMACPP_MIN_P,
        CONF_LLAMACPP_REPEAT_PENALTY,
        CONF_LLAMACPP_PRESENCE_PENALTY,
    ]
    assert all(field in schema for field in expected_fields)


def _validator() -> vol.Schema:
    return vol.Schema(_get_llama_cpp_schema())


class TestValidation:
    """Validation tests for _get_llama_cpp_schema."""

    def test_valid_data_passes(self):
        validator = _validator()
        data = {
            CONF_LLAMACPP_THINKING_MODE: CONF_LLAMACPP_THINKING_MODE_ALWAYS,
            CONF_LLAMACPP_INCLUDE_PRIOR_THINKING: True,
            CONF_LLAMACPP_ID_SLOT: 2,
            CONF_LLAMACPP_TOP_P: 0.5,
            CONF_LLAMACPP_TOP_K: 100,
            CONF_LLAMACPP_MIN_P: 0.1,
            CONF_LLAMACPP_REPEAT_PENALTY: 1.0,
            CONF_LLAMACPP_PRESENCE_PENALTY: 0.0,
        }
        expected = {**data, CONF_LLAMACPP_USE_LOADED_MODEL: False}
        assert validator(data) == expected

    def test_default_thinking_mode_unset(self):
        validator = _validator()
        result = validator({})
        assert CONF_LLAMACPP_THINKING_MODE not in result
        assert result[CONF_LLAMACPP_INCLUDE_PRIOR_THINKING] is True

    def test_thinking_mode_accepts_valid_values(self):
        validator = _validator()
        for value in (
            CONF_LLAMACPP_THINKING_MODE_ALWAYS,
            CONF_LLAMACPP_THINKING_MODE_FIRST_LOOP,
        ):
            result = validator({CONF_LLAMACPP_THINKING_MODE: value})
            assert result[CONF_LLAMACPP_THINKING_MODE] == value

    def test_thinking_mode_rejects_invalid_value(self):
        validator = _validator()
        with pytest.raises(vol.Invalid):
            validator({CONF_LLAMACPP_THINKING_MODE: "invalid"})

    def test_rejects_non_bool_include_prior_reasoning(self):
        validator = _validator()
        with pytest.raises(vol.Invalid):
            validator(
                {
                    CONF_LLAMACPP_THINKING_MODE: CONF_LLAMACPP_THINKING_MODE_ALWAYS,
                    CONF_LLAMACPP_INCLUDE_PRIOR_THINKING: "yes",
                }
            )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            (CONF_LLAMACPP_TOP_P, -0.1),
            (CONF_LLAMACPP_TOP_P, 1.1),
            (CONF_LLAMACPP_TOP_K, 0),
            (CONF_LLAMACPP_TOP_K, -1),
            (CONF_LLAMACPP_MIN_P, -0.1),
            (CONF_LLAMACPP_MIN_P, 1.1),
            (CONF_LLAMACPP_REPEAT_PENALTY, -2.1),
            (CONF_LLAMACPP_REPEAT_PENALTY, 2.1),
            (CONF_LLAMACPP_PRESENCE_PENALTY, -2.1),
            (CONF_LLAMACPP_PRESENCE_PENALTY, 2.1),
        ],
    )
    def test_rejects_out_of_range(self, field, value):
        validator = _validator()
        with pytest.raises(vol.Invalid):
            validator(
                {
                    CONF_LLAMACPP_THINKING_MODE: CONF_LLAMACPP_THINKING_MODE_ALWAYS,
                    CONF_LLAMACPP_INCLUDE_PRIOR_THINKING: True,
                    field: value,
                }
            )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            (CONF_LLAMACPP_TOP_P, 0),
            (CONF_LLAMACPP_TOP_P, 1),
            (CONF_LLAMACPP_TOP_K, 1),
            (CONF_LLAMACPP_TOP_K, 1000),
            (CONF_LLAMACPP_MIN_P, 0),
            (CONF_LLAMACPP_MIN_P, 1),
            (CONF_LLAMACPP_REPEAT_PENALTY, -2),
            (CONF_LLAMACPP_REPEAT_PENALTY, 2),
            (CONF_LLAMACPP_PRESENCE_PENALTY, -2),
            (CONF_LLAMACPP_PRESENCE_PENALTY, 2),
        ],
    )
    def test_accepts_boundary_values(self, field, value):
        validator = _validator()
        data = {
            CONF_LLAMACPP_THINKING_MODE: CONF_LLAMACPP_THINKING_MODE_ALWAYS,
            CONF_LLAMACPP_INCLUDE_PRIOR_THINKING: True,
            field: value,
        }
        validator(data)

    @pytest.mark.parametrize(
        "field",
        [
            CONF_LLAMACPP_TOP_P,
            CONF_LLAMACPP_TOP_K,
            CONF_LLAMACPP_MIN_P,
            CONF_LLAMACPP_REPEAT_PENALTY,
            CONF_LLAMACPP_PRESENCE_PENALTY,
        ],
    )
    def test_rejects_non_numeric_string(self, field):
        validator = _validator()
        with pytest.raises(vol.Invalid):
            validator(
                {
                    CONF_LLAMACPP_THINKING_MODE: CONF_LLAMACPP_THINKING_MODE_ALWAYS,
                    CONF_LLAMACPP_INCLUDE_PRIOR_THINKING: True,
                    field: "abc",
                }
            )
