"""Tests for decision model request and response handling."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

import pytest
from homeassistant.components import ai_task
from homeassistant.components.conversation import Attachment
from homeassistant.exceptions import HomeAssistantError

from custom_components.local_openai.evaluation import (
    build_systemone_request,
    parse_systemone_response,
    read_images,
)

if TYPE_CHECKING:
    from pathlib import Path

QUESTIONS: dict[str, Any] = {
    "delivered": {"type": "noul", "instructions": "Has the package arrived?"},
    "location": {
        "type": "choice",
        "instructions": "Where was the package left?",
        "criteria": {
            "front_door": "At the front door",
            "mailbox": "Inside the mailbox",
        },
    },
    "attention": {
        "type": "score",
        "instructions": "How soon does this need attention?",
        "criteria": ["Informational", "Today", "Right now"],
    },
}

# Response shape returned by llama.cpp for QUESTIONS
SYSTEMONE_RESPONSE: dict[str, Any] = {
    "model": "openjev",
    "answers": {
        "delivered": {"type": "noul", "noul": 0.97},
        "location": {
            "type": "choice",
            "choice": "front_door",
            "probabilities": {"front_door": 0.92, "mailbox": 0.08},
            "confidence": 0.84,
        },
        "attention": {
            "type": "score",
            "score": 0.15,
            "legend": {"0": "Informational", "1": "Today", "2": "Right now"},
            "probabilities": {"0": 0.9, "1": 0.05, "2": 0.05},
            "confidence": 0.8,
        },
    },
    "usage": {"input_tokens": 120, "output_tokens": 0},
}


def _task(
    state: Any = "Your package was left at the front door.",
) -> ai_task.EvaluationTask:
    """Create an evaluation task for QUESTIONS."""
    return ai_task.EvaluationTask(
        name="Check delivery",
        questions=QUESTIONS,
        state=state,
    )


def _response_with_answer(question_id: str, answer: Any) -> dict[str, Any]:
    """Return SYSTEMONE_RESPONSE with one answer replaced."""
    return {
        **SYSTEMONE_RESPONSE,
        "answers": {**SYSTEMONE_RESPONSE["answers"], question_id: answer},
    }


class TestBuildSystemoneRequest:
    """Tests for build_systemone_request."""

    def test_questions_and_state_pass_through(self) -> None:
        """Test the model, state and questions are sent unchanged."""
        assert build_systemone_request("openjev", _task(), []) == {
            "model": "openjev",
            "state": "Your package was left at the front door.",
            "questions": QUESTIONS,
        }

    def test_json_state_is_sent_as_json(self) -> None:
        """Test structured state is sent as JSON rather than a string."""
        state = {"binary_sensor.front_door": "on"}
        assert build_systemone_request("openjev", _task(state), [])["state"] == state

    def test_missing_state_is_sent_empty(self) -> None:
        """Test image-only tasks send an empty state instead of null."""
        assert build_systemone_request("openjev", _task(None), [])["state"] == ""

    def test_images_are_included(self) -> None:
        """Test images are sent when the task has attachments."""
        images = ["data:image/png;base64,AAAA"]
        assert build_systemone_request("openjev", _task(), images)["images"] == images


class TestParseSystemoneResponse:
    """Tests for parse_systemone_response."""

    def test_parses_all_question_types(self) -> None:
        """Test each answer is converted to its Home Assistant answer type."""
        result = parse_systemone_response(_task(), SYSTEMONE_RESPONSE)
        assert result.answers == {
            "delivered": ai_task.NoulAnswer(noul=0.97),
            "location": ai_task.ChoiceAnswer(
                choice="front_door",
                probabilities={"front_door": 0.92, "mailbox": 0.08},
            ),
            "attention": ai_task.ScoreAnswer(
                score=0.15,
                probabilities=[0.9, 0.05, 0.05],
            ),
        }

    def test_score_probabilities_as_list(self) -> None:
        """Test score probabilities already given as a list are accepted."""
        response = _response_with_answer(
            "attention",
            {"type": "score", "score": 1.0, "probabilities": [0.25, 0.5, 0.25]},
        )
        result = parse_systemone_response(_task(), response)
        assert result.answers["attention"] == ai_task.ScoreAnswer(
            score=1.0,
            probabilities=[0.25, 0.5, 0.25],
        )

    def test_answer_without_type_is_accepted(self) -> None:
        """Test answers are matched to their question when the type is omitted."""
        response = _response_with_answer("delivered", {"noul": 0.5})
        result = parse_systemone_response(_task(), response)
        assert result.answers["delivered"] == ai_task.NoulAnswer(noul=0.5)

    def test_extra_answers_are_ignored(self) -> None:
        """Test answers for questions that were not asked are dropped."""
        response = _response_with_answer("unasked", {"type": "noul", "noul": 0.5})
        result = parse_systemone_response(_task(), response)
        assert set(result.answers) == set(QUESTIONS)

    @pytest.mark.parametrize("response", [{}, {"answers": None}, None, "error"])
    def test_missing_answers_raises(self, response: Any) -> None:
        """Test a response without answers raises."""
        with pytest.raises(HomeAssistantError, match="did not contain any answers"):
            parse_systemone_response(_task(), response)

    def test_unanswered_question_raises(self) -> None:
        """Test a question without an answer raises."""
        answers = dict(SYSTEMONE_RESPONSE["answers"])
        del answers["location"]
        with pytest.raises(
            HomeAssistantError, match="did not answer question location"
        ):
            parse_systemone_response(_task(), {"answers": answers})

    @pytest.mark.parametrize(
        ("question_id", "answer"),
        [
            ("delivered", {"type": "choice", "choice": "front_door"}),
            ("delivered", {"type": "noul"}),
            ("delivered", {"type": "noul", "noul": "likely"}),
            ("delivered", 0.97),
            (
                "location",
                {
                    "type": "choice",
                    "choice": "porch",
                    "probabilities": {"front_door": 0.5, "mailbox": 0.5},
                },
            ),
            (
                "location",
                {
                    "type": "choice",
                    "choice": "front_door",
                    "probabilities": {"front_door": 1.0},
                },
            ),
            (
                "attention",
                {"type": "score", "score": 1.0, "probabilities": [0.5, 0.5]},
            ),
            (
                "attention",
                {"type": "score", "score": 1.0, "probabilities": {"0": 0.5, "1": 0.5}},
            ),
        ],
    )
    def test_invalid_answer_raises(self, question_id: str, answer: Any) -> None:
        """Test answers that do not match their question raise."""
        with pytest.raises(
            HomeAssistantError,
            match=f"invalid answer for question {question_id}",
        ):
            parse_systemone_response(
                _task(), _response_with_answer(question_id, answer)
            )


class TestReadImages:
    """Tests for read_images."""

    def test_encodes_images_as_data_urls(self, tmp_path: Path) -> None:
        """Test image attachments are encoded as data URLs."""
        image = tmp_path / "snapshot.jpg"
        image.write_bytes(b"\xff\xd8\xff\xe0jpeg")
        attachment = Attachment(
            media_content_id="media-source://camera/camera.front_door",
            mime_type="image/jpeg",
            path=image,
        )
        encoded = base64.b64encode(b"\xff\xd8\xff\xe0jpeg").decode()
        assert read_images([attachment]) == [f"data:image/jpeg;base64,{encoded}"]

    def test_rejects_non_image_attachments(self, tmp_path: Path) -> None:
        """Test attachments other than images raise."""
        document = tmp_path / "manual.pdf"
        document.write_bytes(b"%PDF")
        attachment = Attachment(
            media_content_id="media-source://media_source/local/manual.pdf",
            mime_type="application/pdf",
            path=document,
        )
        with pytest.raises(HomeAssistantError, match="application/pdf"):
            read_images([attachment])
