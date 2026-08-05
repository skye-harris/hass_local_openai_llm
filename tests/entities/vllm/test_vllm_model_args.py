"""Unit tests for _get_extra_body_args instance method."""

from __future__ import annotations

import pytest

from custom_components.local_openai.const import (
    CONF_VLLM_CONFIG,
    CONF_VLLM_THINKING_TOKEN_BUDGET,
)
from custom_components.local_openai.entities.vllm import VllmMixin

# Stands in for LocalAiEntity, whose _get_extra_body_args builds chat_template_kwargs
BASE_EXTRA_BODY_ARGS = {"chat_template_kwargs": {"enable_thinking": True}}


class _StubBase:
    def _get_extra_body_args(self, options: dict) -> dict:
        return dict(BASE_EXTRA_BODY_ARGS)


class _StubVllmEntity(VllmMixin, _StubBase):
    pass


class TestVllmExtraBodyArgs:
    """Tests for _get_extra_body_args instance method."""

    @pytest.mark.parametrize(
        "options,extra_expected",
        [
            # Budget set
            (
                {CONF_VLLM_CONFIG: {CONF_VLLM_THINKING_TOKEN_BUDGET: 2048}},
                {"thinking_token_budget": 2048},
            ),
            # Float budget converted to int
            (
                {CONF_VLLM_CONFIG: {CONF_VLLM_THINKING_TOKEN_BUDGET: 512.0}},
                {"thinking_token_budget": 512},
            ),
            # Zero is a valid budget, not "unset"
            (
                {CONF_VLLM_CONFIG: {CONF_VLLM_THINKING_TOKEN_BUDGET: 0}},
                {"thinking_token_budget": 0},
            ),
            # No extra options
            ({CONF_VLLM_CONFIG: {}}, {}),
            # None config
            ({}, {}),
            # Budget must be read from the vLLM section, not the top level
            ({CONF_VLLM_THINKING_TOKEN_BUDGET: 2048}, {}),
        ],
    )
    def test_vllm_extra_body_args(self, options: dict, extra_expected: dict) -> None:
        """Test extra body arguments generation with various configurations."""
        result = _StubVllmEntity()._get_extra_body_args(options)
        assert result == BASE_EXTRA_BODY_ARGS | extra_expected
