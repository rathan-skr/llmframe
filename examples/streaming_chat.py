"""
Streaming response demo — prints tokens as they arrive from Claude.

Run:
    python examples/streaming_chat.py
"""
import asyncio

from llmframe import FrameworkConfig, AnthropicProvider, Message


async def main():
    config = FrameworkConfig.from_env()
    provider = AnthropicProvider(api_key=config.anthropic_api_key)

    messages = [
        Message(role="system", content="You are a concise technical writer."),
        Message(role="user", content="Write a 3-paragraph intro to vector databases for a developer audience."),
    ]

    print("Streaming from Claude Opus 4.8:\n")
    print("-" * 60)

    async for chunk in provider.stream(messages):
        print(chunk, end="", flush=True)

    print("\n" + "-" * 60)
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
