# Benchmark API

Every conversation agent has an HTTP endpoint for running automated tests, so you can compare how well different
models and prompts handle voice requests. A request runs one turn through the agent exactly as a real request would,
with the same prompt and tools, but **devices are never controlled**.

## How it works


- Tools that Home Assistant marks as read only, such as `GetLiveContext`, date/time, calendar and to-do lookups, run for
  real, so the model sees the actual state of your home.
- Every other tool call, such as turning on a light, starting a timer or running a script, is recorded and answered
  with a success response without being executed. Calls with invalid arguments still fail as they would for real.
- Pass an `area` to make the request come from a voice satellite in that area. A virtual satellite device is created
  in the area the first time it is used, so the agent is told which area it is in and is offered timer tools, like a
  Voice PE or ESPHome satellite. These devices are removed when the integration reloads or Home Assistant restarts.

## Usage

Requests require a [long-lived access token](https://www.home-assistant.io/docs/authentication/#your-account-profile)
of an admin user. `GET /api/local_openai/benchmark` lists the conversation agents with their endpoints, and the areas
you can use:

```shell
curl -H "Authorization: Bearer $TOKEN" http://homeassistant.local:8123/api/local_openai/benchmark
```

Then send a request to an agent's endpoint:

```shell
curl -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"text": "turn on the lights", "area": "kitchen"}' \
  http://homeassistant.local:8123/api/local_openai/benchmark/<subentry_id>
```

### Request

| Field             | Description                                                                                                  |
|-------------------|--------------------------------------------------------------------------------------------------------------|
| `text`            | The request, as it would be transcribed from speech.                                                         |
| `area`            | Optional area ID or name. When omitted the request has no satellite, like typing in the Home Assistant app. |
| `conversation_id` | Optional ID from a previous response, to test follow up requests. Only benchmark conversations can be continued. |
| `language`        | Optional language, defaults to your Home Assistant language.                                                 |
| `tool_results`    | Optional results to return for tools instead of their default behaviour, keyed by tool name with or without its namespace. Each result can be any JSON value, such as `{"HassTurnOn": {"error": "device offline"}}` or `{"get_weather_forecast": "Sunny, high of 22"}` for tools that return plain text. Useful to pin `GetLiveContext` to a fixed state so results can be reproduced. |

### Response

The response contains the spoken response, every tool call the model made with its arguments and result, and the
timings and token usage of each LLM request:

```json
{
  "conversation_id": "01KA...",
  "agent_id": "conversation.qwen3",
  "area_id": "kitchen",
  "device_id": "4f6c...",
  "speech": "Turned on the kitchen lights.",
  "response_type": "action_done",
  "error_code": null,
  "continue_conversation": false,
  "tool_calls": [
    {
      "iteration": 1,
      "name": "intent__HassTurnOn",
      "arguments": {"area": "Kitchen", "domain": ["light"]},
      "executed": false,
      "mocked": false,
      "error": false,
      "result": {"speech": {}, "response_type": "action_done", "data": {"success": [], "failed": []}}
    }
  ],
  "iterations": [
    {
      "iteration": 1,
      "model": "qwen3",
      "content": null,
      "tools_called": ["intent__HassTurnOn"],
      "finish_reason": "tool_calls",
      "time_to_first_token_ms": 212.4,
      "duration_ms": 640.1,
      "usage": {"prompt_tokens": 1520, "completion_tokens": 24, "total_tokens": 1544},
      "timings": null
    },
    {
      "iteration": 2,
      "model": "qwen3",
      "content": "Turned on the kitchen lights.",
      "tools_called": [],
      "finish_reason": "stop",
      "time_to_first_token_ms": 98.7,
      "duration_ms": 311.9,
      "usage": {"prompt_tokens": 1580, "completion_tokens": 9, "total_tokens": 1589},
      "timings": null
    }
  ],
  "duration_ms": 1203.5
}
```

## Limitations

- Actions always report success without naming the devices they would have affected, so a model that targets a device
  that does not exist is not told so. Grade the tool call arguments rather than the spoken response.
- Timers are not actually started, so a follow up such as "how long is left" will not find them unless you provide a
  result with `tool_results`.
- Requests to agents on the same server run one at a time, so timings are not skewed by other benchmark requests.
- Benchmark requests also fire the `local_openai_llm_generation_complete` event.
