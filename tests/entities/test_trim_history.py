"""Unit tests for _trim_history round-based trimming."""

from __future__ import annotations

from custom_components.local_openai.entity import LocalAiEntity


def _msg(role: str, content: str = "", **kwargs) -> dict:
    """Create a mock message dict."""
    msg: dict = {"role": role, "content": content}
    msg.update(kwargs)
    return msg


def _tool_msg(call_id: str = "call_1", content: str = "result") -> dict:
    """Create a tool result message."""
    return {"role": "tool", "tool_call_id": call_id, "content": content}


class TestTrimHistory:
    """Tests for LocalAiEntity._trim_history round-based trimming."""

    def test_none_keeps_all(self):
        """max_messages=None → keeps all messages."""
        messages = [_msg("system"), _msg("user", "a"), _msg("assistant", "b")]
        result = LocalAiEntity._trim_history(messages, None)
        assert result == messages

    def test_negative_keeps_all(self):
        """max_messages=-1 (defensive) → keeps all messages."""
        messages = [_msg("system"), _msg("user", "a"), _msg("assistant", "b")]
        result = LocalAiEntity._trim_history(messages, -1)
        assert result == messages

    def test_zero_system_user_only(self):
        """max_messages=0 with system + user → keeps only system + user."""
        messages = [_msg("system"), _msg("user", "hello")]
        result = LocalAiEntity._trim_history(messages, 0)
        assert result == [_msg("system"), _msg("user", "hello")]

    def test_zero_keeps_current_round(self):
        """max_messages=0 with system + user + assistant → keeps system + current round only."""
        messages = [
            _msg("system"),
            _msg("user", "q1"),
            _msg("assistant", "a1"),
            _msg("user", "q2"),
            _msg("assistant", "a2"),
        ]
        result = LocalAiEntity._trim_history(messages, 0)
        assert result == [_msg("system"), _msg("user", "q2"), _msg("assistant", "a2")]

    def test_zero_no_system(self):
        """max_messages=0 with no system, just user → keeps user message."""
        messages = [_msg("user", "hello")]
        result = LocalAiEntity._trim_history(messages, 0)
        assert result == [_msg("user", "hello")]

    def test_one_keeps_last_previous_round(self):
        """max_messages=1 with 2 previous rounds → keeps system + last previous round + current."""
        messages = [
            _msg("system"),
            _msg("user", "q1"),
            _msg("assistant", "a1"),
            _msg("user", "q2"),
            _msg("assistant", "a2"),
            _msg("user", "q3"),
            _msg("assistant", "a3"),
        ]
        result = LocalAiEntity._trim_history(messages, 1)
        assert result == [
            _msg("system"),
            _msg("user", "q2"),
            _msg("assistant", "a2"),
            _msg("user", "q3"),
            _msg("assistant", "a3"),
        ]

    def test_tool_call_round_never_split(self):
        """Tool call round (assistant→tool→tool→user) stays intact when trimming."""
        messages = [
            _msg("system"),
            _msg("user", "q1"),
            _msg("assistant", "a1", tool_calls="call_1"),
            _tool_msg("call_1"),
            _msg("user", "q2"),
            _msg("assistant", "a2", tool_calls="call_2"),
            _tool_msg("call_2"),
            _msg("user", "q3"),
            _msg("assistant", "a3"),
        ]
        result = LocalAiEntity._trim_history(messages, 1)
        # The tool call round (q2, assistant(tool_calls), tool result) must stay intact
        assert result == [
            _msg("system"),
            _msg("user", "q2"),
            _msg("assistant", "a2", tool_calls="call_2"),
            _tool_msg("call_2"),
            _msg("user", "q3"),
            _msg("assistant", "a3"),
        ]

    def test_multiple_tool_call_rounds(self):
        """Multiple rounds with tool calls — each round kept/dropped whole."""
        messages = [
            _msg("system"),
            _msg("user", "q1"),
            _msg("assistant", "a1", tool_calls="call_1"),
            _tool_msg("call_1"),
            _msg("user", "q2"),
            _msg("assistant", "a2", tool_calls="call_2"),
            _tool_msg("call_2"),
            _msg("user", "q3"),
            _msg("assistant", "a3", tool_calls="call_3"),
            _tool_msg("call_3"),
            _msg("user", "q4"),
            _msg("assistant", "a4"),
        ]
        result = LocalAiEntity._trim_history(messages, 1)
        assert result == [
            _msg("system"),
            _msg("user", "q3"),
            _msg("assistant", "a3", tool_calls="call_3"),
            _tool_msg("call_3"),
            _msg("user", "q4"),
            _msg("assistant", "a4"),
        ]

    def test_single_round_keeps_all(self):
        """Single round → keep all."""
        messages = [_msg("system"), _msg("user", "q1"), _msg("assistant", "a1")]
        result = LocalAiEntity._trim_history(messages, 1)
        assert result == messages

    def test_empty_messages(self):
        """Empty messages → returns empty."""
        result = LocalAiEntity._trim_history([], 5)
        assert result == []

    def test_single_message(self):
        """Single message → returns as-is."""
        messages = [_msg("system")]
        result = LocalAiEntity._trim_history(messages, 0)
        assert result == messages

    def test_zero_system_only(self):
        """max_messages=0 with system only → returns system alone."""
        messages = [_msg("system")]
        result = LocalAiEntity._trim_history(messages, 0)
        assert result == [_msg("system")]
