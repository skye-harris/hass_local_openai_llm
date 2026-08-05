"""Unit tests for vLLM entity mixin and schema functions."""

from __future__ import annotations

import voluptuous as vol
import pytest

from custom_components.local_openai.const import (
    CONF_VLLM_THINKING_TOKEN_BUDGET,
)
from custom_components.local_openai.entities.vllm import _get_vllm_schema


def test_get_vllm_schema_returns_expected_fields() -> None:
    """Test that schema includes expected vLLM field."""
    schema = _get_vllm_schema()
    assert CONF_VLLM_THINKING_TOKEN_BUDGET in schema


def _validator() -> vol.Schema:
    return vol.Schema(_get_vllm_schema())


class TestValidation:
    """Validation tests for _get_vllm_schema."""

    def test_valid_data_passes(self) -> None:
        validator = _validator()
        data = {CONF_VLLM_THINKING_TOKEN_BUDGET: 2048}
        assert validator(data) == data

    def test_optional_field_can_be_omitted(self) -> None:
        validator = _validator()
        result = validator({})
        assert result == {}

    def test_rejects_non_numeric_string(self) -> None:
        validator = _validator()
        with pytest.raises(vol.Invalid):
            validator({CONF_VLLM_THINKING_TOKEN_BUDGET: "abc"})

    @pytest.mark.parametrize(
        "value",
        [
            -1,
            -100,
            32769,
            100000,
        ],
    )
    def test_rejects_out_of_range(self, value) -> None:
        validator = _validator()
        with pytest.raises(vol.Invalid):
            validator({CONF_VLLM_THINKING_TOKEN_BUDGET: value})

    @pytest.mark.parametrize(
        "value",
        [
            0,
            32768,
        ],
    )
    def test_accepts_boundary_values(self, value) -> None:
        validator = _validator()
        validator({CONF_VLLM_THINKING_TOKEN_BUDGET: value})
