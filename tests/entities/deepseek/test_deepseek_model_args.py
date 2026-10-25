"""Unit tests for _get_extra_body_args instance method."""

from __future__ import annotations

import pytest
from custom_components.local_openai.const import (
    CONF_DEEPSEEK_CONFIG,
    CONF_DEEPSEEK_REASONING_EFFORT,
    CONF_DEEPSEEK_REASONING_EFFORT_DEFAULT,
    CONF_DEEPSEEK_THINKING_MODE,
    CONF_DEEPSEEK_THINKING_MODE_ALWAYS,
    CONF_DEEPSEEK_THINKING_MODE_FIRST_LOOP,
)
from custom_components.local_openai.entities.deepseek import DeepSeekMixin
from custom_components.local_openai.entity import AgentContext

# Stands in for LocalAiEntity, whose _get_extra_body_args builds chat_template_kwargs
BASE_EXTRA_BODY_ARGS = {"chat_template_kwargs": {"hello": "world"}}


class _StubBase:
    def __init__(self, options: dict | None = None) -> None:
        self.options = options or {}

    def _get_extra_body_args(self, context: AgentContext) -> None:
        context.extra_body.update(BASE_EXTRA_BODY_ARGS)


class _StubDeepSeekEntity(DeepSeekMixin, _StubBase):
    pass


class TestDeepSeekExtraBodyArgs:
    """Tests for _get_extra_body_args instance method."""

    def test_super_call_retained(self) -> None:
        """Test that data from the super call is retained in the result."""
        context = AgentContext()
        _StubDeepSeekEntity()._get_extra_body_args(context)
        result = context.extra_body
        assert result["chat_template_kwargs"]["hello"] == "world"

    @pytest.mark.parametrize(
        "options,extra_expected",
        [
            # thinking_mode absent -> thinking disabled
            (
                {CONF_DEEPSEEK_CONFIG: {CONF_DEEPSEEK_REASONING_EFFORT: "high"}},
                {"thinking": {"type": "disabled"}},
            ),
            # thinking_mode = always -> thinking enabled, reasoning_effort included
            (
                {
                    CONF_DEEPSEEK_CONFIG: {
                        CONF_DEEPSEEK_REASONING_EFFORT: "high",
                        CONF_DEEPSEEK_THINKING_MODE: CONF_DEEPSEEK_THINKING_MODE_ALWAYS,
                    }
                },
                {"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
            ),
            # thinking_mode = always with max reasoning_effort
            (
                {
                    CONF_DEEPSEEK_CONFIG: {
                        CONF_DEEPSEEK_REASONING_EFFORT: "max",
                        CONF_DEEPSEEK_THINKING_MODE: CONF_DEEPSEEK_THINKING_MODE_ALWAYS,
                    }
                },
                {"thinking": {"type": "enabled"}, "reasoning_effort": "max"},
            ),
            # thinking_mode = first_loop, iteration=0 -> thinking enabled
            (
                {
                    CONF_DEEPSEEK_CONFIG: {
                        CONF_DEEPSEEK_REASONING_EFFORT: "high",
                        CONF_DEEPSEEK_THINKING_MODE: CONF_DEEPSEEK_THINKING_MODE_FIRST_LOOP,
                    }
                },
                {"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
            ),
            # empty config dict -> thinking disabled
            (
                {CONF_DEEPSEEK_CONFIG: {}},
                {"thinking": {"type": "disabled"}},
            ),
            # no config at all -> thinking disabled
            (
                {},
                {"thinking": {"type": "disabled"}},
            ),
        ],
    )
    def test_deepseek_extra_body_args(
        self, options: dict, extra_expected: dict
    ) -> None:
        """Test extra body arguments generation with various configurations."""
        context = AgentContext()
        _StubDeepSeekEntity(options)._get_extra_body_args(context)
        result = context.extra_body
        assert result == BASE_EXTRA_BODY_ARGS | extra_expected

    @pytest.mark.parametrize(
        "options,iteration,extra_expected",
        [
            # first_loop iteration 0 -> enabled
            (
                {
                    CONF_DEEPSEEK_CONFIG: {
                        CONF_DEEPSEEK_REASONING_EFFORT: "high",
                        CONF_DEEPSEEK_THINKING_MODE: CONF_DEEPSEEK_THINKING_MODE_FIRST_LOOP,
                    }
                },
                0,
                {"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
            ),
            # first_loop iteration 1 -> disabled
            (
                {
                    CONF_DEEPSEEK_CONFIG: {
                        CONF_DEEPSEEK_REASONING_EFFORT: "high",
                        CONF_DEEPSEEK_THINKING_MODE: CONF_DEEPSEEK_THINKING_MODE_FIRST_LOOP,
                    }
                },
                1,
                {"thinking": {"type": "disabled"}},
            ),
            # first_loop iteration 2 -> disabled
            (
                {
                    CONF_DEEPSEEK_CONFIG: {
                        CONF_DEEPSEEK_REASONING_EFFORT: "high",
                        CONF_DEEPSEEK_THINKING_MODE: CONF_DEEPSEEK_THINKING_MODE_FIRST_LOOP,
                    }
                },
                2,
                {"thinking": {"type": "disabled"}},
            ),
            # always iteration 1 -> still enabled
            (
                {
                    CONF_DEEPSEEK_CONFIG: {
                        CONF_DEEPSEEK_REASONING_EFFORT: "high",
                        CONF_DEEPSEEK_THINKING_MODE: CONF_DEEPSEEK_THINKING_MODE_ALWAYS,
                    }
                },
                1,
                {"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
            ),
        ],
    )
    def test_deepseek_extra_body_args_iteration(
        self, options: dict, iteration: int, extra_expected: dict
    ) -> None:
        """Test extra body arguments generation with iteration parameter."""
        context = AgentContext(iteration=iteration)
        _StubDeepSeekEntity(options)._get_extra_body_args(context)
        result = context.extra_body
        assert result == BASE_EXTRA_BODY_ARGS | extra_expected


class _StubBaseWithReasoning(_StubBase):
    """Simulate request body parameters injecting reasoning_effort."""

    def _get_extra_body_args(self, context: AgentContext) -> None:
        super()._get_extra_body_args(context)
        context.extra_body["reasoning_effort"] = "low"


class _StubDeepSeekEntityWithReasoning(DeepSeekMixin, _StubBaseWithReasoning):
    pass


class TestDeepSeekReasoningEffort:
    """Tests for reasoning_effort handling."""

    def test_reasoning_effort_defaults_when_missing(self) -> None:
        """Test a missing reasoning_effort falls back to the default."""
        context = AgentContext()
        _StubDeepSeekEntity(
            {
                CONF_DEEPSEEK_CONFIG: {
                    CONF_DEEPSEEK_THINKING_MODE: CONF_DEEPSEEK_THINKING_MODE_ALWAYS,
                }
            }
        )._get_extra_body_args(context)

        assert (
            context.extra_body["reasoning_effort"]
            == CONF_DEEPSEEK_REASONING_EFFORT_DEFAULT
        )

    @pytest.mark.parametrize(
        "options,iteration",
        [
            ({}, 0),
            (
                {
                    CONF_DEEPSEEK_CONFIG: {
                        CONF_DEEPSEEK_THINKING_MODE: (
                            CONF_DEEPSEEK_THINKING_MODE_FIRST_LOOP
                        ),
                    }
                },
                1,
            ),
        ],
    )
    def test_reasoning_effort_removed_when_thinking_disabled(
        self,
        options: dict,
        iteration: int,
    ) -> None:
        """Test reasoning_effort is removed when thinking is disabled."""
        context = AgentContext(iteration=iteration)
        _StubDeepSeekEntityWithReasoning(options)._get_extra_body_args(context)

        assert context.extra_body["thinking"] == {"type": "disabled"}
        assert "reasoning_effort" not in context.extra_body
