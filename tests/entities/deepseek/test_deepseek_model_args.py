"""Unit tests for _get_extra_body_args instance method."""

from __future__ import annotations

import pytest
from custom_components.local_openai.const import (
    CONF_DEEPSEEK_CONFIG,
    CONF_DEEPSEEK_REASONING_EFFORT,
)
from custom_components.local_openai.entities.deepseek import DeepSeekMixin

# Stands in for LocalAiEntity, whose _get_extra_body_args builds chat_template_kwargs
BASE_EXTRA_BODY_ARGS = {
    "chat_template_kwargs": {"enable_thinking": True, "hello": "world"}
}


class _StubBase:
    def _get_extra_body_args(self, options: dict) -> dict:
        return dict(BASE_EXTRA_BODY_ARGS)


class _StubDeepSeekEntity(DeepSeekMixin, _StubBase):
    pass


class TestDeepSeekExtraBodyArgs:
    """Tests for _get_extra_body_args instance method."""

    def test_super_call_retained(self) -> None:
        """Test that data from the super call is retained in the result."""
        result = _StubDeepSeekEntity()._get_extra_body_args({})
        assert result["chat_template_kwargs"]["hello"] == "world"
        assert result["chat_template_kwargs"]["enable_thinking"] is True

    @pytest.mark.parametrize(
        "options,extra_expected",
        [
            # reasoning_effort set -> thinking enabled
            (
                {CONF_DEEPSEEK_CONFIG: {CONF_DEEPSEEK_REASONING_EFFORT: "high"}},
                {"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
            ),
            # reasoning_effort set to max
            (
                {CONF_DEEPSEEK_CONFIG: {CONF_DEEPSEEK_REASONING_EFFORT: "max"}},
                {"thinking": {"type": "enabled"}, "reasoning_effort": "max"},
            ),
            # Empty config -> thinking disabled, no reasoning_effort
            ({CONF_DEEPSEEK_CONFIG: {}}, {"thinking": {"type": "disabled"}}),
            # None config -> thinking disabled, no reasoning_effort
            ({}, {"thinking": {"type": "disabled"}}),
            # reasoning_effort at top level, not in config section -> ignored
            (
                {CONF_DEEPSEEK_REASONING_EFFORT: "high"},
                {"thinking": {"type": "disabled"}},
            ),
        ],
    )
    def test_deepseek_extra_body_args(
        self, options: dict, extra_expected: dict
    ) -> None:
        """Test extra body arguments generation with various configurations."""
        result = _StubDeepSeekEntity()._get_extra_body_args(options)
        assert result == BASE_EXTRA_BODY_ARGS | extra_expected
