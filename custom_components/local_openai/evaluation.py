"""Decision model support for AI Task evaluate actions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from homeassistant.components import ai_task
from homeassistant.exceptions import HomeAssistantError

from .entity import b64_file

if TYPE_CHECKING:
    from homeassistant.components.conversation import Attachment

# TypeSafe-compatible endpoint served by llama.cpp and Ollama, relative to the base URL
SYSTEMONE_PATH = "/systemone"


def read_images(attachments: list[Attachment]) -> list[str]:
    """Encode image attachments as data URLs, must be run in the executor."""
    images = []
    for attachment in attachments:
        if not attachment.mime_type.startswith("image/"):
            msg = f"Unsupported attachment type for decision models: {attachment.mime_type}"
            raise HomeAssistantError(msg)
        images.append(
            f"data:{attachment.mime_type};base64,{b64_file(attachment.path)}",
        )
    return images


def build_systemone_request(
    model: str,
    task: ai_task.EvaluationTask,
    images: list[str],
) -> dict[str, Any]:
    """Build the /v1/systemone request body for an evaluation task."""
    body: dict[str, Any] = {
        "model": model,
        # Most decision models reject a null state, so image-only tasks send an empty one
        "state": "" if task.state is None else task.state,
        # Home Assistant question definitions follow the TypeSafe API, so they pass through as-is
        "questions": task.questions,
    }
    if images:
        body["images"] = images
    return body


def parse_systemone_response(
    task: ai_task.EvaluationTask,
    response: Any,
) -> ai_task.EvaluationTaskResult:
    """Convert a /v1/systemone response into an answer for every task question."""
    answers = response.get("answers") if isinstance(response, dict) else None
    if not isinstance(answers, dict):
        msg = "Decision model response did not contain any answers"
        raise HomeAssistantError(msg)

    result: dict[str, ai_task.EvaluationAnswer] = {}
    for question_id, question in task.questions.items():
        if question_id not in answers:
            msg = f"Decision model did not answer question {question_id}"
            raise HomeAssistantError(msg)
        try:
            result[question_id] = _parse_answer(question, answers[question_id])
        except (AttributeError, KeyError, TypeError, ValueError) as err:
            msg = (
                f"Decision model returned an invalid answer for question {question_id}"
            )
            raise HomeAssistantError(msg) from err

    return ai_task.EvaluationTaskResult(answers=result)


def _parse_answer(
    question: ai_task.EvaluationQuestion,
    answer: dict[str, Any],
) -> ai_task.EvaluationAnswer:
    """Convert a single /v1/systemone answer, validating it against its question."""
    if answer.get("type", question["type"]) != question["type"]:
        msg = f"Expected a {question['type']} answer, got {answer['type']}"
        raise ValueError(msg)

    if question["type"] == "noul":
        return ai_task.NoulAnswer(noul=float(answer["noul"]))

    if question["type"] == "choice":
        probabilities = {
            option: float(answer["probabilities"][option])
            for option in question["criteria"]
        }
        if answer["choice"] not in probabilities:
            msg = f"Unknown choice {answer['choice']}"
            raise ValueError(msg)
        return ai_task.ChoiceAnswer(
            choice=answer["choice"],
            probabilities=probabilities,
        )

    # Score probabilities are keyed by level index, Home Assistant expects a list
    levels = range(len(question["criteria"]))
    probabilities = answer["probabilities"]
    if isinstance(probabilities, dict):
        probabilities = [probabilities[str(level)] for level in levels]
    if len(probabilities) != len(levels):
        msg = "Score probabilities do not match the number of levels"
        raise ValueError(msg)
    return ai_task.ScoreAnswer(
        score=float(answer["score"]),
        probabilities=[float(probability) for probability in probabilities],
    )
