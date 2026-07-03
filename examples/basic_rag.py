"""
Basic RAG demo: ingest a text snippet, then query it with Claude.

Run:
    python examples/basic_rag.py
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.llmframe import FrameworkConfig, AnthropicProvider, RAGPipeline

SAMPLE = """
FastAPI is a modern, fast web framework for building APIs with Python based on standard
Python type hints. It is built on top of Starlette and Pydantic.

Key features:
- Very high performance, on par with NodeJS and Go (thanks to Starlette and Pydantic).
- Fast to code: Increase the speed to develop features by about 200%-300%.
- Fewer bugs: Reduce about 40% of human induced errors.
- Intuitive: Great editor support. Completion everywhere. Less time debugging.
- Easy: Designed to be easy to use and learn. Less time reading docs.
- Short: Minimize code duplication. Multiple features from each parameter declaration.
- Robust: Get production-ready code. With automatic interactive documentation.
- Standards-based: Based on and fully compatible with the open standards for APIs.

FastAPI also supports asynchronous request handlers, dependency injection,
OAuth2 with JWT tokens, and automatic OpenAPI + JSON Schema documentation.
"""


async def main():
    config = FrameworkConfig.from_env()

    if not config.anthropic_api_key:
        print("ERROR: Set ANTHROPIC_API_KEY in your .env file.")
        return
    if not config.openai_api_key:
        print("ERROR: Set OPENAI_API_KEY in your .env file (used for embeddings).")
        return

    llm = AnthropicProvider(api_key=config.anthropic_api_key)
    rag = RAGPipeline(llm=llm, config=config)

    print("Ingesting document...")
    doc = await rag.ingest(SAMPLE, source="fastapi-overview")
    print(f"  Stored document: {doc.id}")

    questions = [
        "What makes FastAPI fast?",
        "Does FastAPI support async?",
        "What standards does FastAPI follow?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        result = await rag.query(q)
        print(f"A: {result.answer}")
        print(f"   ({result.input_tokens} in / {result.output_tokens} out tokens, {len(result.sources)} sources)")

    # Clean up (optional)
    await rag.delete_document(doc.id)
    print("\nDone. Document removed from vector store.")


if __name__ == "__main__":
    asyncio.run(main())
