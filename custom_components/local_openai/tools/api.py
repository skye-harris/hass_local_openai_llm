"""HomeAssistant API instance to wrap our stop_talking tool."""

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.llm import selector_serializer

from custom_components.local_openai.tools.stop_talking import StopTalkingTool


class InternalToolApi(llm.API):
    """Internal tools API, not registered externally."""

    def __init__(
        self,
        hass: HomeAssistant,
        name: str,
    ) -> None:
        """Initialize the API."""
        super().__init__(hass=hass, id=id or name.lower().replace(" ", "_"), name=name)

    async def async_get_api_instance(
        self,
        llm_context: llm.LLMContext,
    ) -> llm.APIInstance:
        """Get API instance."""
        return llm.APIInstance(
            api=self,
            api_prompt="",
            llm_context=llm_context,
            tools=[StopTalkingTool()],
            custom_serializer=selector_serializer,
        )
