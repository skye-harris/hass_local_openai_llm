"""Tests for API prompt injection into the AI Task system prompt."""

from __future__ import annotations

from types import MappingProxyType
from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.components import ai_task, conversation
from homeassistant.components.conversation import ChatLog, SystemContent
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import CONF_LLM_HASS_API, CONF_MODEL
from homeassistant.core import HomeAssistant

from custom_components.local_openai.ai_task import LocalAITaskEntity
from custom_components.local_openai.const import CONF_AI_TASK_TOOLS_SECTION

API_PROMPT = "Use the provided tools to help with light control."


def _make_entity(hass: HomeAssistant) -> LocalAITaskEntity:
    """Create an AI Task entity with LLM APIs configured."""
    entry = MagicMock(spec=ConfigEntry)
    entry.data = {}
    entry.runtime_data = MagicMock()
    subentry = ConfigSubentry(
        subentry_id="test_ai_task_subentry_id",
        subentry_type="ai_task_data",
        title="AI Task",
        data=MappingProxyType(
            {
                CONF_MODEL: "test-model",
                CONF_AI_TASK_TOOLS_SECTION: {CONF_LLM_HASS_API: ["test_llm_api"]},
            }
        ),
        unique_id=None,
    )
    entity = LocalAITaskEntity(entry, subentry)
    entity.hass = hass
    return entity


def _make_chat_log(hass: HomeAssistant, system_prompt: str) -> ChatLog:
    """Create a chat log with a system prompt like HA's async_provide_llm_data."""
    return ChatLog(
        hass=hass,
        conversation_id="test_conversation_id",
        content=[SystemContent(content=system_prompt)],
    )


async def _run_generate_data(
    hass: HomeAssistant,
    system_prompt: str,
) -> ChatLog:
    entity = _make_entity(hass)

    apis = MagicMock()
    apis.api_prompt = f"  {API_PROMPT}  "
    apis.custom_serializer = None

    async def fake_handle_chat_log(**kwargs) -> None:
        kwargs["chat_log"].content.append(
            conversation.AssistantContent(agent_id="test_agent", content="done")
        )

    entity._async_handle_chat_log = AsyncMock(side_effect=fake_handle_chat_log)

    with patch(
        "custom_components.local_openai.ai_task.async_get_api",
        AsyncMock(return_value=apis),
    ):
        task = ai_task.GenDataTask(name="Test", instructions="Do the thing")
        chat_log = _make_chat_log(hass, system_prompt)
        await entity._async_generate_data(task, chat_log)

    return chat_log


async def test_api_prompt_goes_before_current_time(hass: HomeAssistant) -> None:
    """The API prompt is injected just before the Current time line."""
    system_prompt = (
        "You are a Home Assistant expert.\nCurrent time is 10:00:00. "
        "Today's date is 2026-11-26.\n"
    )

    chat_log = await _run_generate_data(hass, system_prompt)

    first = chat_log.content[0]
    assert isinstance(first, SystemContent)
    assert API_PROMPT in first.content
    assert "Current time is 10:00:00" in first.content
    assert first.content.index(API_PROMPT) < first.content.index("Current time is")


async def test_api_prompt_is_stripped_and_separated(hass: HomeAssistant) -> None:
    """The API prompt is stripped and surrounded by newlines before Current time."""
    system_prompt = "You are a Home Assistant expert.\nCurrent time is 10:00:00.\n"

    chat_log = await _run_generate_data(hass, system_prompt)

    first = chat_log.content[0]
    assert isinstance(first, SystemContent)
    expected = (
        f"You are a Home Assistant expert.\n\n{API_PROMPT}\n\nCurrent time is 10:00:00."
    )
    assert first.content.startswith(expected)


async def test_api_prompt_replaced_not_appended(hass: HomeAssistant) -> None:
    """The full prompt contains the api prompt exactly once."""
    system_prompt = "Base prompt\nCurrent time is 10:00:00.\n"

    chat_log = await _run_generate_data(hass, system_prompt)

    first = chat_log.content[0]
    assert isinstance(first, SystemContent)
    assert first.content.count(API_PROMPT) == 1
