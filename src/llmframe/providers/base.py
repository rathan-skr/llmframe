from __future__ import annotations
from abc import ABC, abstractmethod
from typing import AsyncIterator
from ..core.types import Message, ChatResponse


class BaseLLMProvider(ABC):
    """Abstract base for all LLM providers — swap Claude, GPT, or any model."""

    @abstractmethod
    async def chat(self, messages: list[Message], **kwargs) -> ChatResponse:
        """Single non-streaming completion."""
        ...

    @abstractmethod
    async def stream(self, messages: list[Message], **kwargs) -> AsyncIterator[str]:
        """Streaming completion — yields text chunks as they arrive."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...

    @property
    @abstractmethod
    def default_model(self) -> str:
        ...
