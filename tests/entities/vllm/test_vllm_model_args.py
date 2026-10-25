"""Unit tests for _get_extra_body_args instance method."""

from __future__ import annotations

import pytest
from custom_components.local_openai.const import (
    CONF_VLLM_CONFIG,
    CONF_VLLM_THINKING_MODE,
    CONF_VLLM_THINKING_MODE_ALWAYS,
    CONF_VLLM_THINKING_MODE_FIRST_LOOP,
    CONF_VLLM_THINKING_TOKEN_BUDGET,
)
from custom_components.local_openai.entities.vllm import VllmMixin
from custom_components.local_openai.entity import AgentContext

# Stands in for LocalAiEntity, whose _get_extra_body_args builds chat_template_kwargs
BASE_EXTRA_BODY_ARGS = {"chat_template_kwargs": {"hello": "world"}}


class _StubBase:
    def __init__(self, options: dict | None = None) -> None:
        self.options = options or {}

    def _get_extra_body_args(self, context: AgentContext) -> None:
        context.extra_body.update(BASE_EXTRA_BODY_ARGS)


class _StubVllmEntity(VllmMixin, _StubBase):
    pass


class TestVllmExtraBodyArgs:
    """Tests for _get_extra_body_args instance method."""

    def test_super_call_retained(self) -> None:
        """Test that data from the super call is retained in the result."""
        context = AgentContext()
        _StubVllmEntity()._get_extra_body_args(context)
        result = context.extra_body
        assert result["chat_template_kwargs"]["hello"] == "world"

    @pytest.mark.parametrize(
        "options,extra_expected",
        [
            # No thinking_mode at all -> enable_thinking = False
            (
                {CONF_VLLM_CONFIG: {}},
                {"chat_template_kwargs": {"hello": "world", "enable_thinking": False}},
            ),
            # thinking_mode = always -> enable_thinking = True, budget uses default
            (
                {
                    CONF_VLLM_CONFIG: {
                        CONF_VLLM_THINKING_MODE: CONF_VLLM_THINKING_MODE_ALWAYS,
                    }
                },
                {
                    "chat_template_kwargs": {
                        "hello": "world",
                        "enable_thinking": True,
                    },
                    "thinking_token_budget": 1024,
                },
            ),
            # thinking_mode = first_loop, iteration=0 -> enable_thinking = True, budget uses default
            (
                {
                    CONF_VLLM_CONFIG: {
                        CONF_VLLM_THINKING_MODE: CONF_VLLM_THINKING_MODE_FIRST_LOOP,
                    }
                },
                {
                    "chat_template_kwargs": {
                        "hello": "world",
                        "enable_thinking": True,
                    },
                    "thinking_token_budget": 1024,
                },
            ),
            # budget only, no thinking_mode -> thinking disabled, budget omitted
            (
                {
                    CONF_VLLM_CONFIG: {
                        CONF_VLLM_THINKING_TOKEN_BUDGET: 2048,
                    }
                },
                {"chat_template_kwargs": {"hello": "world", "enable_thinking": False}},
            ),
            # budget + always -> enable_thinking = True + budget
            (
                {
                    CONF_VLLM_CONFIG: {
                        CONF_VLLM_THINKING_MODE: CONF_VLLM_THINKING_MODE_ALWAYS,
                        CONF_VLLM_THINKING_TOKEN_BUDGET: 2048,
                    }
                },
                {
                    "chat_template_kwargs": {
                        "hello": "world",
                        "enable_thinking": True,
                    },
                    "thinking_token_budget": 2048,
                },
            ),
            # Float budget converted to int
            (
                {
                    CONF_VLLM_CONFIG: {
                        CONF_VLLM_THINKING_MODE: CONF_VLLM_THINKING_MODE_ALWAYS,
                        CONF_VLLM_THINKING_TOKEN_BUDGET: 512.0,
                    }
                },
                {
                    "chat_template_kwargs": {
                        "hello": "world",
                        "enable_thinking": True,
                    },
                    "thinking_token_budget": 512,
                },
            ),
            # Zero budget is valid
            (
                {
                    CONF_VLLM_CONFIG: {
                        CONF_VLLM_THINKING_MODE: CONF_VLLM_THINKING_MODE_ALWAYS,
                        CONF_VLLM_THINKING_TOKEN_BUDGET: 0,
                    }
                },
                {
                    "chat_template_kwargs": {
                        "hello": "world",
                        "enable_thinking": True,
                    },
                    "thinking_token_budget": 0,
                },
            ),
            # None config
            (
                {},
                {"chat_template_kwargs": {"hello": "world", "enable_thinking": False}},
            ),
            # Budget must be read from the vLLM section, not the top level
            (
                {CONF_VLLM_THINKING_TOKEN_BUDGET: 2048},
                {"chat_template_kwargs": {"hello": "world", "enable_thinking": False}},
            ),
        ],
    )
    def test_vllm_extra_body_args(self, options: dict, extra_expected: dict) -> None:
        """Test extra body arguments generation with various configurations."""
        context = AgentContext()
        _StubVllmEntity(options)._get_extra_body_args(context)
        result = context.extra_body
        assert result == extra_expected

    @pytest.mark.parametrize(
        "options,iteration,extra_expected",
        [
            # first_loop iteration 0 -> enabled, budget uses default
            (
                {
                    CONF_VLLM_CONFIG: {
                        CONF_VLLM_THINKING_MODE: CONF_VLLM_THINKING_MODE_FIRST_LOOP,
                    }
                },
                0,
                {
                    "chat_template_kwargs": {
                        "hello": "world",
                        "enable_thinking": True,
                    },
                    "thinking_token_budget": 1024,
                },
            ),
            # first_loop iteration 1 -> disabled
            (
                {
                    CONF_VLLM_CONFIG: {
                        CONF_VLLM_THINKING_MODE: CONF_VLLM_THINKING_MODE_FIRST_LOOP,
                    }
                },
                1,
                {"chat_template_kwargs": {"hello": "world", "enable_thinking": False}},
            ),
            # first_loop iteration 2 -> disabled
            (
                {
                    CONF_VLLM_CONFIG: {
                        CONF_VLLM_THINKING_MODE: CONF_VLLM_THINKING_MODE_FIRST_LOOP,
                    }
                },
                2,
                {"chat_template_kwargs": {"hello": "world", "enable_thinking": False}},
            ),
            # always iteration 1 -> still enabled, budget uses default
            (
                {
                    CONF_VLLM_CONFIG: {
                        CONF_VLLM_THINKING_MODE: CONF_VLLM_THINKING_MODE_ALWAYS,
                    }
                },
                1,
                {
                    "chat_template_kwargs": {
                        "hello": "world",
                        "enable_thinking": True,
                    },
                    "thinking_token_budget": 1024,
                },
            ),
            # always iteration 5 -> still enabled, budget uses default
            (
                {
                    CONF_VLLM_CONFIG: {
                        CONF_VLLM_THINKING_MODE: CONF_VLLM_THINKING_MODE_ALWAYS,
                    }
                },
                5,
                {
                    "chat_template_kwargs": {
                        "hello": "world",
                        "enable_thinking": True,
                    },
                    "thinking_token_budget": 1024,
                },
            ),
            # first_loop iteration 0 with budget -> budget included
            (
                {
                    CONF_VLLM_CONFIG: {
                        CONF_VLLM_THINKING_MODE: CONF_VLLM_THINKING_MODE_FIRST_LOOP,
                        CONF_VLLM_THINKING_TOKEN_BUDGET: 2048,
                    }
                },
                0,
                {
                    "chat_template_kwargs": {
                        "hello": "world",
                        "enable_thinking": True,
                    },
                    "thinking_token_budget": 2048,
                },
            ),
            # first_loop iteration 1 with budget -> budget removed
            (
                {
                    CONF_VLLM_CONFIG: {
                        CONF_VLLM_THINKING_MODE: CONF_VLLM_THINKING_MODE_FIRST_LOOP,
                        CONF_VLLM_THINKING_TOKEN_BUDGET: 2048,
                    }
                },
                1,
                {"chat_template_kwargs": {"hello": "world", "enable_thinking": False}},
            ),
            # always iteration 1 with budget -> budget retained
            (
                {
                    CONF_VLLM_CONFIG: {
                        CONF_VLLM_THINKING_MODE: CONF_VLLM_THINKING_MODE_ALWAYS,
                        CONF_VLLM_THINKING_TOKEN_BUDGET: 2048,
                    }
                },
                1,
                {
                    "chat_template_kwargs": {
                        "hello": "world",
                        "enable_thinking": True,
                    },
                    "thinking_token_budget": 2048,
                },
            ),
        ],
    )
    def test_vllm_extra_body_args_iteration(
        self, options: dict, iteration: int, extra_expected: dict
    ) -> None:
        """Test extra body arguments generation with iteration parameter."""
        context = AgentContext(iteration=iteration)
        _StubVllmEntity(options)._get_extra_body_args(context)
        result = context.extra_body
        assert result == extra_expected


class _StubBaseWithBudget(_StubBase):
    """Simulate request body parameters injecting thinking_token_budget."""

    def _get_extra_body_args(self, context: AgentContext) -> None:
        super()._get_extra_body_args(context)
        context.extra_body["thinking_token_budget"] = 999


class _StubVllmEntityWithBudget(VllmMixin, _StubBaseWithBudget):
    pass


def test_request_body_budget_removed_when_thinking_disabled() -> None:
    """Test a request-body thinking budget is removed when thinking is disabled."""
    context = AgentContext()
    _StubVllmEntityWithBudget()._get_extra_body_args(context)

    assert "thinking_token_budget" not in context.extra_body
    assert context.extra_body["chat_template_kwargs"]["enable_thinking"] is False
