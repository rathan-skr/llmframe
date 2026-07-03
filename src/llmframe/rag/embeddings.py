from __future__ import annotations
from abc import ABC, abstractmethod
from ..core.types import Chunk


class BaseEmbeddingProvider(ABC):
    @abstractmethod
    async def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]: ...

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]: ...


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return chunks
        response = await self._client.embeddings.create(
            model=self.model,
            input=[c.content for c in chunks],
        )
        for chunk, data in zip(chunks, response.data):
            chunk.embedding = data.embedding
        return chunks

    async def embed_query(self, query: str) -> list[float]:
        response = await self._client.embeddings.create(model=self.model, input=[query])
        return response.data[0].embedding


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """Runs sentence-transformers locally — no API key, no cost, no network call."""

    def __init__(self, model: str = "BAAI/bge-large-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model)

    async def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return chunks
        embeddings = self._model.encode(
            [c.content for c in chunks], normalize_embeddings=True
        ).tolist()
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
        return chunks

    async def embed_query(self, query: str) -> list[float]:
        return self._model.encode([query], normalize_embeddings=True)[0].tolist()


# backward-compatible alias
EmbeddingProvider = OpenAIEmbeddingProvider


def create_embedding_provider(
    provider: str, api_key: str = "", model: str = ""
) -> BaseEmbeddingProvider:
    """Factory — set EMBEDDING_PROVIDER=local or openai in .env to switch."""
    if provider == "local":
        return LocalEmbeddingProvider(model=model or "BAAI/bge-large-en-v1.5")
    return OpenAIEmbeddingProvider(api_key=api_key, model=model or "text-embedding-3-small")
