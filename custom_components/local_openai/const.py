"""Constants for the Local OpenAI LLM integration."""

import logging

from homeassistant.const import CONF_LLM_HASS_API, CONF_PROMPT
from homeassistant.helpers import llm

DOMAIN = "local_openai"
LOGGER = logging.getLogger(__package__)

CONF_RECOMMENDED = "recommended"
CONF_BASE_URL = "base_url"
CONF_SERVER_NAME = "server_name"
CONF_STRIP_EMOJIS = "strip_emojis"
CONF_MAX_MESSAGE_HISTORY = "max_message_history"
CONF_TEMPERATURE = "temperature"
CONF_PARALLEL_TOOL_CALLS = "parallel_tool_calls"

CONF_WEAVIATE_OPTIONS = "weaviate_options"
CONF_WEAVIATE_HOST = "weaviate_host"
CONF_WEAVIATE_API_KEY = "weaviate_api_key"
CONF_WEAVIATE_CLASS_NAME = "weaviate_class_name"
CONF_WEAVIATE_MAX_RESULTS = "weaviate_max_results"
CONF_WEAVIATE_THRESHOLD = "weaviate_threshold"

CONF_WEAVIATE_DEFAULT_CLASS_NAME = "Homeassistant"
CONF_WEAVIATE_DEFAULT_THRESHOLD = 0.7
CONF_WEAVIATE_DEFAULT_MAX_RESULTS = 2

RECOMMENDED_CONVERSATION_OPTIONS = {
    CONF_RECOMMENDED: True,
    CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
}
