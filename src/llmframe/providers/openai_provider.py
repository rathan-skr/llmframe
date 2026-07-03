from __future__ import annotations
from openai import AsyncOpenAI
from typing import AsyncIterator
from ..core.types import Message, ChatResponse
from ..core.config import LLMConfig
from .base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """GPT-4o provider — secondary LLM for comparison or cost optimization."""

    def __init__(self, api_key: str, config: LLMConfig | None = None):
        self._client = AsyncOpenAI(api_key=api_key)
        self._config = config or LLMConfig(provider="openai", model="gpt-4o")

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def default_model(self) -> str:
        return "gpt-4o"

    def _to_api_messages(self, messages: list[Message]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    async def chat(self, messages: list[Message], **kwargs) -> ChatResponse:
        model = kwargs.get("model", self._config.model)
        max_tokens = kwargs.get("max_tokens", self._config.max_tokens)

        response = await self._client.chat.completions.create(
            model=model,
            messages=self._to_api_messages(messages),
            max_tokens=max_tokens,
        )

        choice = response.choices[0]
        return ChatResponse(
            content=choice.message.content or "",
            model=model,
            provider=self.provider_name,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )

    async def stream(self, messages: list[Message], **kwargs) -> AsyncIterator[str]:
        model = kwargs.get("model", self._config.model)
        max_tokens = kwargs.get("max_tokens", self._config.max_tokens)

        stream = await self._client.chat.completions.create(
            model=model,
            messages=self._to_api_messages(messages),
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
