"""Unit tests for _get_model_args and _get_extra_body_args instance methods."""

from __future__ import annotations

import pytest

from custom_components.local_openai.const import (
    CONF_LLAMACPP_CONFIG,
    CONF_LLAMACPP_ENABLE_THINKING,
    CONF_LLAMACPP_ID_SLOT,
    CONF_LLAMACPP_MIN_P,
    CONF_LLAMACPP_PRESENCE_PENALTY,
    CONF_LLAMACPP_REPEAT_PENALTY,
    CONF_LLAMACPP_TOP_K,
    CONF_LLAMACPP_TOP_P,
)
from custom_components.local_openai.entities.llama_cpp import LlamaCppMixin


class TestLlamaCppModelArgs:
    """Tests for _get_model_args instance method."""

    @pytest.mark.parametrize(
        "options,sampling_params,expected",
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
                ["top_p", "top_k", "min_p", "repeat_penalty", "presence_penalty"],
                {
                    "top_p": 0.9,
                    "top_k": 50,
                    "min_p": 0.05,
                    "repeat_penalty": 1.1,
                    "presence_penalty": 0.5,
                },
            ),
            # Partial params
            (
                {
                    CONF_LLAMACPP_CONFIG: {
                        CONF_LLAMACPP_TOP_P: 0.85,
                        CONF_LLAMACPP_TOP_K: 100,
                    }
                },
                ["top_p", "top_k"],
                {"top_p": 0.85, "top_k": 100},
            ),
            # No sampling params
            ({CONF_LLAMACPP_CONFIG: {}}, [], {}),
            # None config
            ({}, [], {}),
        ],
    )
    def test_llama_cpp_model_args_with_params(self, options, sampling_params, expected):
        """Test sampling parameter conversion with various inputs."""
        result = LlamaCppMixin()._get_model_args(options)
        assert result == expected

    @pytest.mark.parametrize(
        "options,expected",
        [
            (
                {CONF_LLAMACPP_CONFIG: {CONF_LLAMACPP_TOP_K: 50.0}},
                {"top_k": 50},
            ),
            (
                {CONF_LLAMACPP_CONFIG: {CONF_LLAMACPP_TOP_P: 1}},
                {"top_p": 1.0},
            ),
            (
                {CONF_LLAMACPP_CONFIG: {CONF_LLAMACPP_MIN_P: 0}},
                {"min_p": 0.0},
            ),
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
    def test_llama_cpp_model_args_type_conversion(self, options, expected):
        result = LlamaCppMixin()._get_model_args(options)
        assert result == expected
        for key, value in expected.items():
            assert isinstance(result[key], type(value))


class TestLlamaCppExtraBodyArgs:
    """Tests for _get_extra_body_args instance method."""

    @pytest.mark.parametrize(
        "options,expected",
        [
            # Both id_slot and enable_thinking set
            (
                {
                    CONF_LLAMACPP_CONFIG: {
                        CONF_LLAMACPP_ID_SLOT: 2,
                        CONF_LLAMACPP_ENABLE_THINKING: True,
                    }
                },
                {
                    "id_slot": 2,
                    "chat_template_kwargs": {"enable_thinking": True},
                },
            ),
            # Only enable_thinking set
            (
                {
                    CONF_LLAMACPP_CONFIG: {
                        CONF_LLAMACPP_ENABLE_THINKING: False,
                    }
                },
                {
                    "chat_template_kwargs": {"enable_thinking": False},
                },
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
                    "chat_template_kwargs": {"enable_thinking": False},
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
                        CONF_LLAMACPP_ENABLE_THINKING: True,
                    }
                },
                {
                    "id_slot": 3,
                    "chat_template_kwargs": {"enable_thinking": True},
                },
            ),
            # Int enable_thinking converted to bool
            (
                {
                    CONF_LLAMACPP_CONFIG: {
                        CONF_LLAMACPP_ID_SLOT: 1,
                        CONF_LLAMACPP_ENABLE_THINKING: 0,
                    }
                },
                {
                    "id_slot": 1,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            ),
        ],
    )
    def test_llama_cpp_extra_body_args(self, options: dict, expected: dict):
        """Test extra body arguments generation with various configurations."""
        result = LlamaCppMixin()._get_extra_body_args(options)
        assert result == expected
