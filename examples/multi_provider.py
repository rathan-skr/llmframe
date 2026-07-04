"""
Side-by-side comparison: same question answered by Claude vs GPT-4o.

Run:
    python examples/multi_provider.py
"""
import asyncio
import time

from llmframe import FrameworkConfig, AnthropicProvider, OpenAIProvider, Message


async def ask(provider, question: str):
    messages = [Message(role="user", content=question)]
    t0 = time.perf_counter()
    response = await provider.chat(messages)
    elapsed = time.perf_counter() - t0
    return response, elapsed


async def main():
    config = FrameworkConfig.from_env()
    claude = AnthropicProvider(api_key=config.anthropic_api_key)
    gpt = OpenAIProvider(api_key=config.openai_api_key)

    question = "Explain the CAP theorem in two sentences, with a real-world example."

    print(f"Question: {question}\n")
    print("=" * 60)

    claude_resp, claude_time = await ask(claude, question)
    print(f"Claude Opus 4.8 ({claude_time:.2f}s):")
    print(claude_resp.content)
    print(f"Tokens: {claude_resp.input_tokens} in / {claude_resp.output_tokens} out\n")

    print("=" * 60)

    gpt_resp, gpt_time = await ask(gpt, question)
    print(f"GPT-4o ({gpt_time:.2f}s):")
    print(gpt_resp.content)
    print(f"Tokens: {gpt_resp.input_tokens} in / {gpt_resp.output_tokens} out\n")


if __name__ == "__main__":
    asyncio.run(main())
