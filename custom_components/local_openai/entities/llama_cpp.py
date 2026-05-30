"""Server-specific entities for llama.cpp."""

from custom_components.local_openai.ai_task import LocalAITaskEntity
from custom_components.local_openai.conversation import LocalAiConversationEntity


class LlamaCppConversationEntity(LocalAiConversationEntity):
    """Conversation agent for llama.cpp servers."""


class LlamaCppAITaskEntity(LocalAITaskEntity):
    """AI Task entity for llama.cpp servers."""
