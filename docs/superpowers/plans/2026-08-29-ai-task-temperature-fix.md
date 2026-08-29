# AI Task Temperature Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `ai_task.generate_data` with a `structure` from failing on Claude-5 / reasoning-class models by making `temperature` opt-in, and stop structured tasks whose name contains spaces from 400ing by sanitizing the JSON-schema name.

**Architecture:** Two coordinated behavior changes plus one independent fix. (1) `_async_handle_chat_log` stops emitting a defaulted `temperature`; the key is added to the request body only when the user has explicitly configured it, so unset means the provider applies its own default (which reasoning models require). (2) The AI Task subentry schema gains an Optional `temperature` field, and the Conversation subentry field becomes Optional (no forced default) for consistency. (3) `_format_structured_output` slugifies and length-caps the schema name.

**Tech Stack:** Python 3.13, Home Assistant custom integration, voluptuous schemas, OpenAI async client, pytest / pytest-homeassistant-custom-component.

**Spec:** `bugreport.md` (repo root)

## Global Constraints

- Integration: `custom_components/local_openai` (Local OpenAI LLM).
- `temperature` MUST be omitted entirely from the request body when unset — send neither the key nor `null`. Only omission makes the provider fall back to its own default.
- `temperature` stays a dedicated config field: do NOT remove `CONF_TEMPERATURE` from `REQUEST_BODY_CONFIGURABLE_PARAMETERS` (`config_flow.py:157-164`). It must remain rejected in the "Request Body Parameters" section.
- `response_format.json_schema.name` must match `^[a-zA-Z0-9_-]+$` and be ≤ 64 chars.
- Existing configs that store `temperature: 0.6` must keep sending `0.6` (no silent behavior change for them).
- Run tests with: `pytest` from repo root. Lint with `ruff check` (the repo uses ruff — see recent commit "That was ruff").

---

### Task 1: Make `temperature` opt-in in the request body

**Files:**
- Modify: `custom_components/local_openai/entity.py:683-693`
- Test: `tests/entities/test_entity.py`

**Interfaces:**
- Consumes: `LocalAiEntity.options` (property → `self.subentry.data`, a mapping), `CONF_TEMPERATURE` (already imported in `entity.py:48`).
- Produces: `model_args` dict built inside `_async_handle_chat_log` that contains a `"temperature"` key only when `options.get(CONF_TEMPERATURE)` is not `None`.

- [ ] **Step 1: Write the failing tests**

Add a new test class to `tests/entities/test_entity.py`. It drives `_async_handle_chat_log` with `_run_agent_loop` patched to capture the assembled `model_args`, using a minimal chat log so the message/tool machinery stays out of the way.

```python
from types import MappingProxyType
from unittest.mock import patch

from custom_components.local_openai.const import CONF_TEMPERATURE


class TestTemperatureOptIn:
    """Temperature is only sent when explicitly configured."""

    @staticmethod
    def _make_chat_log() -> MagicMock:
        chat_log = MagicMock(spec=conversation.ChatLog)
        chat_log.content = []
        chat_log.llm_api = None
        chat_log.conversation_id = "conv-1"
        return chat_log

    async def test_temperature_omitted_when_unset(
        self,
        mock_conversation_entity: conversation.ConversationAgent,
    ):
        entity = mock_conversation_entity
        # subentry.data has no CONF_TEMPERATURE
        entity.subentry.data = MappingProxyType({"model": "test-model"})

        captured: dict = {}

        async def fake_loop(client, model_args, *args, **kwargs):
            captured.update(model_args)

        with patch.object(entity, "_run_agent_loop", side_effect=fake_loop):
            await entity._async_handle_chat_log(self._make_chat_log())

        assert "temperature" not in captured

    async def test_temperature_sent_when_set(
        self,
        mock_conversation_entity: conversation.ConversationAgent,
    ):
        entity = mock_conversation_entity
        entity.subentry.data = MappingProxyType(
            {"model": "test-model", CONF_TEMPERATURE: 0.6}
        )

        captured: dict = {}

        async def fake_loop(client, model_args, *args, **kwargs):
            captured.update(model_args)

        with patch.object(entity, "_run_agent_loop", side_effect=fake_loop):
            await entity._async_handle_chat_log(self._make_chat_log())

        assert captured["temperature"] == 0.6
```

Note: `ConfigSubentry.data` is read-only in production, but the test fixture builds a plain `ConfigSubentry` whose `data` attribute is reassignable here; if assignment raises in this environment, rebuild the subentry via `mock_conversation_subentry`'s pattern with the desired `data`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/entities/test_entity.py::TestTemperatureOptIn -v`
Expected: `test_temperature_omitted_when_unset` FAILS — `"temperature"` is present because it is currently hardcoded to the `0.6` default.

- [ ] **Step 3: Implement the opt-in logic**

In `custom_components/local_openai/entity.py`, replace the current block:

```python
        temperature = options.get(CONF_TEMPERATURE, 0.6)

        model_args = {
            "model": self.model,
            "temperature": temperature,
            "parallel_tool_calls": parallel_tool_calls,
            "extra_headers": {
                "HTTP-Referer": "https://github.com/skye-harris/hass_local_openai_llm",
                "X-Title": "Home Assistant",
            },
        }
```

with:

```python
        model_args = {
            "model": self.model,
            "parallel_tool_calls": parallel_tool_calls,
            "extra_headers": {
                "HTTP-Referer": "https://github.com/skye-harris/hass_local_openai_llm",
                "X-Title": "Home Assistant",
            },
        }

        # Only send temperature when explicitly configured. Reasoning models
        # (Claude-5 in thinking mode, triggered by response_format) reject a
        # non-default temperature, so an unset value must be omitted entirely
        # rather than defaulted — letting the provider apply its own default.
        temperature = options.get(CONF_TEMPERATURE)
        if temperature is not None:
            model_args["temperature"] = temperature
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/entities/test_entity.py::TestTemperatureOptIn -v`
Expected: both tests PASS.

- [ ] **Step 5: Run the full entity test file to check for regressions**

Run: `pytest tests/entities/test_entity.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add custom_components/local_openai/entity.py tests/entities/test_entity.py
git commit -m "fix: only send temperature when explicitly configured

Reasoning models (Claude-5 in thinking mode via response_format) reject a
non-default temperature. Omit the key entirely when unset so the provider
applies its own default, fixing structured ai_task.generate_data."
```

---

### Task 2: Add an Optional temperature field to AI Task and relax the Conversation field

**Files:**
- Modify: `custom_components/local_openai/config_flow.py:639-649` (Conversation field)
- Modify: `custom_components/local_openai/config_flow.py:872-915` (`AITaskDataFlowHandler.get_schema`)
- Test: `tests/config_flow/test_config_flow_helpers.py`

**Interfaces:**
- Consumes: `CONF_TEMPERATURE` (already imported in `config_flow.py:109`), `NumberSelector`, `NumberSelectorConfig`, `NumberSelectorMode` (already imported), `AITaskDataFlowHandler.get_schema`.
- Produces: an AI Task subentry schema whose keys include a `vol.Optional(CONF_TEMPERATURE)` marker; a Conversation subentry schema whose `CONF_TEMPERATURE` marker is `vol.Optional` (not `vol.Required`).

- [ ] **Step 1: Write the failing test**

Add to `tests/config_flow/test_config_flow_helpers.py`. This constructs the AI Task handler with a mocked model list and inspects the returned schema's markers.

```python
import voluptuous as vol
from unittest.mock import AsyncMock, MagicMock

from custom_components.local_openai.config_flow import AITaskDataFlowHandler
from custom_components.local_openai.const import CONF_TEMPERATURE


def _marker_for(schema: vol.Schema, key: str):
    """Return the voluptuous Marker whose schema string equals key."""
    for marker in schema.schema:
        if str(marker) == key:
            return marker
    return None


async def test_ai_task_schema_has_optional_temperature(hass) -> None:
    handler = AITaskDataFlowHandler()
    handler.hass = hass

    entry = MagicMock()
    entry.data = {}
    entry.runtime_data = MagicMock()
    entry.runtime_data.models.list = AsyncMock(return_value=MagicMock(data=[]))
    handler._get_entry = MagicMock(return_value=entry)

    schema = await handler.get_schema()

    marker = _marker_for(schema, CONF_TEMPERATURE)
    assert marker is not None, "AI Task schema must expose a temperature field"
    assert isinstance(marker, vol.Optional)
    # Optional with no forced default → unset stays unset.
    assert marker.default is vol.UNDEFINED
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/config_flow/test_config_flow_helpers.py::test_ai_task_schema_has_optional_temperature -v`
Expected: FAIL — the AI Task schema currently has no `temperature` key, so `marker is None`.

- [ ] **Step 3: Add the Optional temperature field to the AI Task schema**

In `AITaskDataFlowHandler.get_schema`, inside the `schema = { ... }` dict (`config_flow.py:872-915`), add this entry after the `CONF_AI_TASK_SUPPORTED_ATTRIBUTES` selector and before `CONF_AI_TASK_TOOLS_SECTION`:

```python
            vol.Optional(
                CONF_TEMPERATURE,
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0,
                    max=1,
                    step=0.01,
                    mode=NumberSelectorMode.BOX,
                ),
            ),
```

- [ ] **Step 4: Relax the Conversation temperature field**

In the Conversation subentry schema (`config_flow.py:639-649`), change:

```python
            vol.Required(
                CONF_TEMPERATURE,
                default=0.6,
            ): NumberSelector(
```

to:

```python
            vol.Optional(
                CONF_TEMPERATURE,
            ): NumberSelector(
```

Leave the `NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.01, mode=NumberSelectorMode.BOX))` body unchanged.

- [ ] **Step 5: Run the new test and the existing helpers suite**

Run: `pytest tests/config_flow/test_config_flow_helpers.py -v`
Expected: the new test PASSES; the existing `test_global_errors[temperature]` (which asserts temperature is rejected under Request Body Parameters) still PASSES — `REQUEST_BODY_CONFIGURABLE_PARAMETERS` is unchanged.

- [ ] **Step 6: Commit**

```bash
git add custom_components/local_openai/config_flow.py tests/config_flow/test_config_flow_helpers.py
git commit -m "feat: expose optional temperature on AI Task, relax Conversation field

AI Task subentry gains an optional Temperature field; the Conversation
field becomes Optional (no forced default) so 'unset' is representable and
consistent. Leaving it empty omits temperature from the request."
```

---

### Task 3: Sanitize the structured-output schema name

**Files:**
- Modify: `custom_components/local_openai/entity.py:122-131` (`_format_structured_output`)
- Modify: `custom_components/local_openai/entity.py:20` (import `slugify`)
- Test: `tests/entities/test_entity.py`

**Interfaces:**
- Consumes: `homeassistant.util.slugify`, `_format_structured_output(name: str, schema: vol.Schema, llm_api) -> JSONSchema` (module-level function).
- Produces: `_format_structured_output` output whose `"name"` matches `^[a-zA-Z0-9_-]+$` and is ≤ 64 chars.

- [ ] **Step 1: Write the failing test**

Add to `tests/entities/test_entity.py`:

```python
import voluptuous as vol

from custom_components.local_openai.entity import _format_structured_output


class TestFormatStructuredOutputName:
    """The json_schema name must be API-safe."""

    def test_spaces_are_slugified(self):
        result = _format_structured_output(
            "my_test task", vol.Schema({}), None
        )
        assert result["name"] == "my_test_task"

    def test_name_capped_at_64_chars(self):
        long_name = "a" * 200
        result = _format_structured_output(long_name, vol.Schema({}), None)
        assert len(result["name"]) <= 64
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest "tests/entities/test_entity.py::TestFormatStructuredOutputName" -v`
Expected: `test_spaces_are_slugified` FAILS — name is currently passed through verbatim (`"my_test task"`).

- [ ] **Step 3: Add the slugify import**

In `custom_components/local_openai/entity.py:20`, change:

```python
from homeassistant.util import dt as dt_util
```

to:

```python
from homeassistant.util import dt as dt_util, slugify
```

- [ ] **Step 4: Sanitize the name in `_format_structured_output`**

In `custom_components/local_openai/entity.py:127-131`, change:

```python
    result: JSONSchema = {
        "name": name,
        "strict": True,
    }
```

to:

```python
    result: JSONSchema = {
        # The API requires json_schema.name to match ^[a-zA-Z0-9_-]+$ and be
        # <= 64 chars; task names may contain spaces/punctuation. Mirror HA
        # core's openai_conversation which slugifies this value.
        "name": slugify(name)[:64],
        "strict": True,
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest "tests/entities/test_entity.py::TestFormatStructuredOutputName" -v`
Expected: both tests PASS.

- [ ] **Step 6: Commit**

```bash
git add custom_components/local_openai/entity.py tests/entities/test_entity.py
git commit -m "fix: slugify and cap structured-output schema name

response_format.json_schema.name must match ^[a-zA-Z0-9_-]+$ and be <= 64
chars; task names with spaces previously 400'd. Slugify like HA core."
```

---

### Task 4: Full suite + lint verification

**Files:** none (verification only).

- [ ] **Step 1: Run the full test suite**

Run: `pytest`
Expected: all tests PASS (no regressions in conversation, request-body-parameters, tool-ordering, or content-injection suites).

- [ ] **Step 2: Run the linter**

Run: `ruff check custom_components/local_openai tests`
Expected: no errors. Fix any (e.g. import ordering on the `slugify` import) and re-run.

- [ ] **Step 3: Commit any lint fixes (if needed)**

```bash
git add -A
git commit -m "chore: ruff"
```

---

## Self-Review

**Spec coverage:**
- Root cause (hardcoded `temperature: 0.6` at `entity.py:683-687`) → Task 1.
- "No user-side workaround": missing AI Task Temperature field (`config_flow.py:872-915`) → Task 2 Step 3; Conversation field forces a default (`config_flow.py:639-649`) → Task 2 Step 4.
- Suggested fix #1 (opt-in, omit key when unset) → Task 1 Step 3, verbatim omission (no `null`).
- Suggested fix #2 (optional fields, model-agnostic) → Task 2; chosen over the "omit only when structured" alternative per decision.
- Secondary bug (unsanitized `json_schema.name`, `entity.py:122-129` / `ai_task.py:154`) → Task 3, fixed at the single choke-point `_format_structured_output` (covers all callers, so `ai_task.py:154` needs no change).
- Constraint "keep temperature rejected in Request Body Parameters" → Task 2 Step 5 asserts the existing `test_global_errors[temperature]` still passes; `REQUEST_BODY_CONFIGURABLE_PARAMETERS` untouched.

**Placeholder scan:** No TBD/TODO; all steps carry concrete code and exact commands.

**Type consistency:** `_format_structured_output(name, schema, llm_api)` signature matches `entity.py:122-126`. `_run_agent_loop(client, model_args, chat_log, strip_emojis, conversation_id=None)` matches the patched call in Task 1's `fake_loop(client, model_args, *args, **kwargs)`. `CONF_TEMPERATURE` import paths match `const.py:19`.

**Migration:** Existing Conversation configs storing `0.6` still send `0.6` (value present → key added). Existing AI Task configs (no temperature key) → omitted → provider default → structured tasks now work. Verified against `options.get(CONF_TEMPERATURE)` semantics in Task 1.
