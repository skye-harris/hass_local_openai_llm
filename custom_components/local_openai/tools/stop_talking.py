"""Stop Talking tool class."""

import probatio
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from custom_components.local_openai.const import DOMAIN, LOGGER, TOOL_NAME_STOP_TALKING


class StopTalkingTool(llm.Tool):
    """Stop Talking tool: used to instruct the harness to NOT continue the conversation."""

    integration = DOMAIN

    name = TOOL_NAME_STOP_TALKING
    description = "Call this tool by itself if the user has requested that you stop talking or 'shut up'"
    parameters = probatio.Schema({})

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> llm.ToolResult:
        """Call the stop talking tool."""
        LOGGER.debug("StopTalkingTool called by LLM")
        return llm.ToolResult(data={})
