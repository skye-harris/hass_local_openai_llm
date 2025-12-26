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
- Parallel tool calling requires support from both your model and inference server.
  - In some cases, control of this is handled by the server directly, in which case toggling this will not have any result.  

## Retrieval Augmented Generation (RAG) with Weaviate

Retrieval Augmented Generation is used to pre-feed your LLM messages with related data to provide contextually relevant information to the model based on the user input message.

This integration supports connecting your Agent to a Weaviate vector database server.
Once configured, user messages to the Agent will be queried against the Weaviate database first, and the result data pre-emptively injected into the current conversation as contextual data for the Agent to utilise in their response. 

### Weaviate Configuration

1. Install Weaviate [locally](https://docs.weaviate.io/weaviate/quickstart/local)
   1. A pre-made `docker-compose.yml` is provided in the `weaviate` directory of this repository.
   2. Weaviate Cloud is not supported as there is no free tier available.
2. Reconfigure your LLM Server entity (**not** the Agent entity) in Home Assistant.
   1. Expand the `Weaviate configuration` section and fill in the details server address and API key (`homeassistant` if using the supplied `docker-compose.yml`).
3. Optional: Reconfigure your AI Agent entities in Home Assistant.
   1. This is only needed if you wish to change the default Weaviate values on a per-agent basis:
      1. Object class name: Defaults to `Homeassistant`, can be changed if you want a different data store for the Agent. The integration will handle creating the required object class within Weaviate if it does not already exist. 
      2. Maximum number of results to use: Defaults to `2`.
      3. Result score threshold: Defaults to `0.7`. Higher requires a closer semantic match, while loser is less strict.

### Notes

- Only the current generations user message is queried in the database, no prior user messages are included

## Additional

Looking to add some more functionality to your Home Assistant conversation agent, such as web and localised business/location search? Check out my [Tools for Assist](https://github.com/skye-harris/llm_intents) integration here!

## Acknowledgements

- This integration is forked from the [OpenRouter](https://github.com/home-assistant/core/tree/dev/homeassistant/components/open_router) integration for Home Assistant by [@joostlek](https://github.com/joostlek)

---

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/skyeharris)
