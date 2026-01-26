# Local OpenAI LLM <small>_(Custom Integration for Home Assistant)_</small>

**Allows use of generic OpenAI-compatible LLM services, such as (but not limited to):**

- llama.cpp
- vLLM
- LM Studio
- Ollama
- OpenRouter
- Scaleway

**This integration has been forked from Home Assistants OpenRouter integration, with the following changes:**

- Added server URL to the initial server configuration
- Made the API Key optional during initial server configuration: can be left blank if your local server does not require one
- Uses streamed LLM responses
- Conversation Agents support TTS streaming
- Automatically strips `<think>` tags from responses
- Added support for image inputs for AI Task Agents
- Added support for reconfiguring Conversation Agents
- Added option to trim conversation history to help stay within your context window
- Added temperature control
- Added option to strip emojis from responses
- Added support for parallel tool calling
- Added experimental Retrieval Augmented Generation capability
- Added chat template arguments support

---

## Installation

### Install via HACS (recommended)

Have [HACS](https://hacs.xyz/) installed, this will allow you to update easily.

Adding Tools for Assist to HACS can be using this button:  
[![image](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=skye-harris&repository=hass_local_openai_llm&category=integration)

<br>

> [!NOTE]
> If the button above doesn't work, add `https://github.com/skye-harris/hass_local_openai_llm` as a custom repository of type Integration in HACS.

* Click install on the `Local OpenAI LLM` integration.
* Restart Home Assistant.

<details><summary>Manual Install</summary>

* Copy the `local_openai`  folder from [latest release](https://github.com/skye-harris/hass_local_openai_llm/releases/latest) to the [
  `custom_components` folder](https://developers.home-assistant.io/docs/creating_integration_file_structure/#where-home-assistant-looks-for-integrations) in your config directory.
* Restart the Home Assistant.

</details>

## Integration Configuration

After installation, configure the integration through Home Assistant's UI:

1. Go to `Settings` → `Devices & Services`.
2. Click `Add Integration`.
3. Search for `Local OpenAI LLM`.
4. Follow the setup wizard to configure your desired services.

### Configuration Notes

- The Server URL must be a fully qualified URL pointing to an OpenAI-compatible API.
    - This typically ends with `/v1` but may differ depending on your server configuration.
- If you have the `Extended OpenAI Conversation` integration installed, this has a dependency of an older version of the OpenAI client library.
    - It is strongly recommended this be uninstalled to ensure that HACS installs the correct OpenAI client library.
- Assist requires a fairly lengthy context for tooling and entity definitions.
    - It is strongly recommended to use _at least_ 8k context size and to limit history length to avoid context overflow issues.
    - This is not configurable through OpenAI-compatible APIs, and needs to be configured with the inference server directly.
- Tool calling must be enabled in your inference engine, eg:
    - **vLLM**: https://docs.vllm.ai/en/latest/features/tool_calling/
    - **llama.cpp**: https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md
- Parallel tool calling requires support from both your model and inference server.
    - In some cases, control of this is handled by the server directly, in which case toggling this will not have any result.
- Chat Template Arguments allow you to provide custom arguments to your model
    - Arguments are supplied as key/value pairs and provided to the `chat_template_kwargs` request parameter
    - Values support Jinja2 templates, in order to provide non-string and more complex data structures
    - Arguments differ per model, and not all models make use of user-provided arguments
    - See your models documentation for what arguments are available to be used

### Experimental: Date/Time Context Injection Role

This integration supports injecting some dynamic content, presently the date and time, into the active Conversation Agent prompt when making a request.
This was added as it is beneficial for the model to be grounded with this context in its role as an assistant, and was previously added to the system prompt by Home Assistant itself before later being removed due to negative effects on prompt caching
and performance.

This was previously always-on but has been extracted as an **experimental** configuration option as this is not a once-size-fits-all for all models.
To this end I have provided a number of options so that users can try them out and select the one that works best, or disable entirely if none work well, for their chosen model.

The available options are:

#### <u>Tool Result</u>:
The date and time are inserted as a `Tool Call Result` message to the model, before the current user message.

As long as the model does not reject it, this is the recommended method to use and produces the most reliable results during testing.

#### <u>Assistant</u>:
The date and time are inserted as an additional `Assistant` message to the model, before the current user message.

In cases where the `Tool Call Result` role method does not work for a model, this is the next recommended to test with.

#### <u>User</u>:
The date and time are inserted as an additional `User` message to the model, before the current user message.

Recommended only where neither the `System` nor `Assistant` injection methods work for the model, but may not produce desirable results.
Some models have been known to repeat the date/time back to the user without request.

#### <u>Disabled (no selection)</u>:
If your model simply refuses to work well with any method, simply remove the value from the configuration option to disable this again.


## Experimental: Retrieval Augmented Generation (RAG) with Weaviate

Retrieval Augmented Generation is used to pre-feed your LLM messages with related data to provide contextually relevant information to the model based on the user input message.

This integration supports connecting your Agent to a Weaviate vector database server.
Once configured, user messages to the Agent will be queried against the Weaviate database first, and the result data pre-emptively injected into the current conversation as contextual data for the Agent to utilise in their response.

This is not a general-purpose "memory" for the Agent: content is only provided to the Agent if it matches on the current user input message to the model.

See the [Weaviate documentation](https://docs.weaviate.io/weaviate) for further information on Weaviate.

### Weaviate Configuration

1. **Install Weaviate [locally](https://docs.weaviate.io/weaviate/quickstart/local)**
    1. A pre-made `docker-compose.yml` is provided in the `weaviate` directory of this repository.
    2. _Weaviate Cloud is not supported: there is no free tier available and its cheapest pricing plan isn't attractive for personal/home use, and so I don't anticipate demand for this._
2. **Reconfigure your LLM Server entity (**not** the Agent entity) in Home Assistant.**
    1. Expand the `Weaviate configuration` section and fill in the details server address and API key (`homeassistant` if using the supplied `docker-compose.yml`).
3. **Optional: Reconfigure your AI Agent entities in Home Assistant.**
    1. This is only needed if you wish to change the default Weaviate values on a per-agent basis:
        1. Object class name: Defaults to `Homeassistant`, can be changed if you want a different data store for the Agent. The integration will handle creating the required object class within Weaviate if it does not already exist.
        2. Maximum number of results to use: Defaults to `2`.
        3. Result score threshold: Defaults to `0.9`.
        4. Hybrid search alpha: Defaults to `0.5`. Balances the hybrid result scoring between 0 (fully text-matched) and 1 (fully vectorised) matching.

### Managing Data

Self-hosted Weaviate does not come with a front-end to manage data at all at this current point in time.

I have included a simple NodeJS-based WebApp server within the `/weaviate` directory of this repository, that can be used to connect to your local Weaviate instance and view, query, and manage the data in your object class.
This is also setup into the supplied `docker-compose.yml` and exposed on port 9090 by default.

This tool supports the following basic functionality:

- Connect to your server and list the available object classes.
- View the available entry data in each class.
- Add new entry data to a class.
- Perform vector and hybrid searches against an object class.

_This is not a general-purpose Weaviate management tool, rather it is purpose-built specifically for use with this integration and the object classes that it creates._

### Notes

- Only the current generations user message is queried in the database, no prior user messages are included.
- Search results are used for the **current** user/assistant turns only (including multiturn tool usages), and do not carry forward to subsequent user/assistant turns.
- Objects are stored as 2 pieces of data: the `query`, and the `content`:
    - The `query` is what is vectorised and the user inputs searched against.
    - The `content` is the main content to be provided to be fed to the LLM, along with its `query` text for context.
- Useful for providing contextual information to the LLM for different types of requests, without having all of it in your prompt at all times.
- I have performed basic testing of this with a variety of models across a few inference providers:
    - Qwen 3-VL 8B locally on llama.cpp.
    - Minimax m2.1 on OpenRouter.
    - Ministral 8B 2512 on OpenRouter.
    - GPT-5 on OpenRouter.
    - Gemma 3 27B on Scaleway.
    - Llama 3.1 8B on Scaleway.
    - GPT-OSS-120B on Scaleway.
- A service action, `local_openai.add_to_weaviate`, can be used from within Home Assistant to add content to the database.

## Web Search & Additional Tools

Looking to add some more functionality to your Home Assistant conversation agent, such as web and localised business/location search? Check out my [Tools for Assist](https://github.com/skye-harris/llm_intents) integration here!

These tools exist as a separate integration for compatibility across the wider Home Assistant Conversation ecosystem.

## Acknowledgements

- This integration is forked from the [OpenRouter](https://github.com/home-assistant/core/tree/dev/homeassistant/components/open_router) integration for Home Assistant by [@joostlek](https://github.com/joostlek)

---

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/skyeharris)
