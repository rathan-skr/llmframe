from __future__ import annotations
import anthropic
from typing import AsyncIterator
from ..core.types import Message, ChatResponse
from ..core.config import LLMConfig
from .base import BaseLLMProvider


class AnthropicProvider(BaseLLMProvider):
    """Claude provider — defaults to claude-opus-4-8 with adaptive thinking and streaming."""

    def __init__(self, api_key: str, config: LLMConfig | None = None):
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._config = config or LLMConfig()

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def default_model(self) -> str:
        return "claude-opus-4-8"

    def _split_system(self, messages: list[Message]) -> tuple[str | None, list[dict]]:
        system = None
        api_msgs = []
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                api_msgs.append({"role": m.role, "content": m.content})
        return system, api_msgs

    async def chat(self, messages: list[Message], **kwargs) -> ChatResponse:
        model = kwargs.get("model", self._config.model)
        max_tokens = kwargs.get("max_tokens", self._config.max_tokens)
        system, api_msgs = self._split_system(messages)

        params: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": api_msgs,
            "thinking": {"type": "adaptive"},
        }
        if system:
            params["system"] = system

        response = await self._client.messages.create(**params)

        thinking_text = None
        answer_text = ""
        for block in response.content:
            if block.type == "thinking":
                thinking_text = getattr(block, "thinking", None)
            elif block.type == "text":
                answer_text += block.text

        return ChatResponse(
            content=answer_text,
            model=model,
            provider=self.provider_name,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            thinking=thinking_text,
        )

    async def stream(self, messages: list[Message], **kwargs) -> AsyncIterator[str]:
        model = kwargs.get("model", self._config.model)
        max_tokens = kwargs.get("max_tokens", self._config.max_tokens)
        system, api_msgs = self._split_system(messages)

        params: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": api_msgs,
            "thinking": {"type": "adaptive"},
        }
        if system:
            params["system"] = system

        async with self._client.messages.stream(**params) as s:
            async for text in s.text_stream:
                yield text
