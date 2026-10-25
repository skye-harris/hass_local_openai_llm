"""Unit tests for _get_extra_body_args instance method."""

from __future__ import annotations

import copy

import pytest
from custom_components.local_openai.const import (
    CONF_LLAMACPP_CONFIG,
    CONF_LLAMACPP_ID_SLOT,
    CONF_LLAMACPP_MIN_P,
    CONF_LLAMACPP_PRESENCE_PENALTY,
    CONF_LLAMACPP_REPEAT_PENALTY,
    CONF_LLAMACPP_THINKING_MODE,
    CONF_LLAMACPP_THINKING_MODE_ALWAYS,
    CONF_LLAMACPP_THINKING_MODE_FIRST_LOOP,
    CONF_LLAMACPP_TOP_K,
    CONF_LLAMACPP_TOP_P,
)
from custom_components.local_openai.entities.llama_cpp import LlamaCppMixin
from custom_components.local_openai.entity import AgentContext

BASE_EXTRA_BODY_ARGS = {
    "chat_template_kwargs": {"hello": "world", "enable_thinking": False},
    "stream_options": {"include_usage": True},
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    result.pop("stream_options", None)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = {**result[key], **value}
        else:
            result[key] = value
    return result


class _StubBase:
    def __init__(self, options: dict | None = None) -> None:
        self.options = options or {}

    def _get_extra_body_args(self, context: AgentContext) -> None:
        context.extra_body.update(copy.deepcopy(BASE_EXTRA_BODY_ARGS))


class _StubLlamaCppEntity(LlamaCppMixin, _StubBase):
    def __init__(self, options: dict | None = None) -> None:
        self.options = options or {}


class TestLlamaCppExtraBodyArgs:
    """Tests for _get_extra_body_args instance method."""

    def test_super_call_retained(self) -> None:
        """Test that data from the super call is retained in the result."""
        context = AgentContext()
        _StubLlamaCppEntity()._get_extra_body_args(context)
        result = context.extra_body
        assert result["chat_template_kwargs"]["hello"] == "world"

    @pytest.mark.parametrize(
        "options,expected",
        [
            # Both id_slot and thinking_mode set
            (
                {
                    CONF_LLAMACPP_CONFIG: {
                        CONF_LLAMACPP_ID_SLOT: 2,
                        CONF_LLAMACPP_THINKING_MODE: CONF_LLAMACPP_THINKING_MODE_ALWAYS,
                    }
                },
                {
                    "id_slot": 2,
                    "chat_template_kwargs": {"enable_thinking": True},
                },
            ),
            # Only thinking_mode set
            (
                {
                    CONF_LLAMACPP_CONFIG: {
                        CONF_LLAMACPP_THINKING_MODE: CONF_LLAMACPP_THINKING_MODE_FIRST_LOOP,
                    }
                },
                {"chat_template_kwargs": {"enable_thinking": True}},
            ),
            # Only id_slot set
            (
                {
                    CONF_LLAMACPP_CONFIG: {
                        CONF_LLAMACPP_ID_SLOT: 1,
                    }
                },
                {
                    "id_slot": 1,
                },
            ),
            # No extra options
            (
                {CONF_LLAMACPP_CONFIG: {}},
                {"chat_template_kwargs": {"enable_thinking": False}},
            ),
            # None config
            ({}, {"chat_template_kwargs": {"enable_thinking": False}}),
            # Float id_slot converted to int
            (
                {
                    CONF_LLAMACPP_CONFIG: {
                        CONF_LLAMACPP_ID_SLOT: 3.0,
                        CONF_LLAMACPP_THINKING_MODE: CONF_LLAMACPP_THINKING_MODE_ALWAYS,
                    }
                },
                {
                    "id_slot": 3,
                    "chat_template_kwargs": {"enable_thinking": True},
                },
            ),
        ],
    )
    def test_llama_cpp_extra_body_args(self, options: dict, expected: dict):
        """Test extra body arguments generation with various configurations."""
        context = AgentContext()
        _StubLlamaCppEntity(options)._get_extra_body_args(context)
        result = context.extra_body
        assert result == _deep_merge(BASE_EXTRA_BODY_ARGS, expected)

    @pytest.mark.parametrize(
        "options,expected",
        [
            # All sampling params set
            (
                {
                    CONF_LLAMACPP_CONFIG: {
                        CONF_LLAMACPP_TOP_P: 0.9,
                        CONF_LLAMACPP_TOP_K: 50,
                        CONF_LLAMACPP_MIN_P: 0.05,
                        CONF_LLAMACPP_REPEAT_PENALTY: 1.1,
                        CONF_LLAMACPP_PRESENCE_PENALTY: 0.5,
                    }
                },
                {
                    "chat_template_kwargs": {
                        "hello": "world",
                        "enable_thinking": False,
                    },
                    "top_p": 0.9,
                    "top_k": 50,
                    "min_p": 0.05,
                    "repeat_penalty": 1.1,
                    "presence_penalty": 0.5,
                },
            ),
            # Partial sampling params
            (
                {
                    CONF_LLAMACPP_CONFIG: {
                        CONF_LLAMACPP_TOP_P: 0.85,
                        CONF_LLAMACPP_TOP_K: 100,
                    }
                },
                {
                    "chat_template_kwargs": {
                        "hello": "world",
                        "enable_thinking": False,
                    },
                    "top_p": 0.85,
                    "top_k": 100,
                },
            ),
            # No sampling params
            (
                {CONF_LLAMACPP_CONFIG: {}},
                {"chat_template_kwargs": {"hello": "world"}},
            ),
            # None config
            ({}, {"chat_template_kwargs": {"hello": "world"}}),
        ],
    )
    def test_llama_cpp_extra_body_args_with_sampling_params(
        self,
        options: dict,
        expected: dict,
    ):
        """Test extra_body with sampling parameters."""
        context = AgentContext()
        _StubLlamaCppEntity(options)._get_extra_body_args(context)
        result = context.extra_body
        assert result == _deep_merge(BASE_EXTRA_BODY_ARGS, expected)

    @pytest.mark.parametrize(
        "options,expected_sampling",
        [
            ({CONF_LLAMACPP_CONFIG: {CONF_LLAMACPP_TOP_K: 50.0}}, {"top_k": 50}),
            ({CONF_LLAMACPP_CONFIG: {CONF_LLAMACPP_TOP_P: 1}}, {"top_p": 1.0}),
            ({CONF_LLAMACPP_CONFIG: {CONF_LLAMACPP_MIN_P: 0}}, {"min_p": 0.0}),
            (
                {CONF_LLAMACPP_CONFIG: {CONF_LLAMACPP_REPEAT_PENALTY: -2}},
                {"repeat_penalty": -2.0},
            ),
            (
                {CONF_LLAMACPP_CONFIG: {CONF_LLAMACPP_PRESENCE_PENALTY: 1}},
                {"presence_penalty": 1.0},
            ),
        ],
    )
    def test_llama_cpp_extra_body_sampling_type_conversion(
        self,
        options: dict,
        expected_sampling: dict,
    ):
        """Test type conversion of sampling parameters in extra_body."""
        context = AgentContext()
        _StubLlamaCppEntity(options)._get_extra_body_args(context)
        result = context.extra_body
        assert result["chat_template_kwargs"]["hello"] == "world"
        for key, value in expected_sampling.items():
            assert result[key] == value
            assert isinstance(result[key], type(value))

    @pytest.mark.parametrize(
        "thinking_mode,iteration,expected_enable_thinking",
        [
            (None, 0, False),
            (None, 1, False),
            (CONF_LLAMACPP_THINKING_MODE_ALWAYS, 0, True),
            (CONF_LLAMACPP_THINKING_MODE_ALWAYS, 3, True),
            (CONF_LLAMACPP_THINKING_MODE_FIRST_LOOP, 0, True),
            (CONF_LLAMACPP_THINKING_MODE_FIRST_LOOP, 1, False),
            (CONF_LLAMACPP_THINKING_MODE_FIRST_LOOP, 2, False),
        ],
    )
    def test_llama_cpp_thinking_mode_iteration(
        self,
        thinking_mode: str | None,
        iteration: int,
        expected_enable_thinking: bool,
    ) -> None:
        """Test thinking is disabled when unset and after the first loop."""
        options = {CONF_LLAMACPP_CONFIG: {}}
        if thinking_mode is not None:
            options[CONF_LLAMACPP_CONFIG][CONF_LLAMACPP_THINKING_MODE] = thinking_mode
        context = AgentContext(iteration=iteration)
        _StubLlamaCppEntity(options)._get_extra_body_args(context)
        assert (
            context.extra_body["chat_template_kwargs"]["enable_thinking"]
            is expected_enable_thinking
        )
