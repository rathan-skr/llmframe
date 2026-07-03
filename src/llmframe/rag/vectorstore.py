from __future__ import annotations
from abc import ABC, abstractmethod
import chromadb
from chromadb.config import Settings
from ..core.types import Chunk, SearchResult


class BaseVectorStore(ABC):
    @abstractmethod
    async def add(self, chunks: list[Chunk]) -> None: ...

    @abstractmethod
    async def search(self, query_embedding: list[float], k: int = 5) -> list[SearchResult]: ...

    @abstractmethod
    async def delete_by_document(self, document_id: str) -> None: ...


class ChromaVectorStore(BaseVectorStore):
    """Persistent local ChromaDB — no external service required for development."""

    def __init__(self, persist_dir: str = "./data/chroma", collection: str = "documents"):
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    async def add(self, chunks: list[Chunk]) -> None:
        missing = [c.id for c in chunks if c.embedding is None]
        if missing:
            raise ValueError(f"Chunks missing embeddings: {missing}")

        self._col.upsert(
            ids=[c.id for c in chunks],
            documents=[c.content for c in chunks],
            embeddings=[c.embedding for c in chunks],  # type: ignore[arg-type]
            metadatas=[c.metadata for c in chunks],
        )

    async def search(self, query_embedding: list[float], k: int = 5) -> list[SearchResult]:
        count = self._col.count()
        if count == 0:
            return []

        results = self._col.query(
            query_embeddings=[query_embedding],
            n_results=min(k, count),
            include=["documents", "metadatas", "distances"],
        )

        out: list[SearchResult] = []
        for rank, (rid, doc, meta, dist) in enumerate(
            zip(
                results["ids"][0],
                results["documents"][0],  # type: ignore[index]
                results["metadatas"][0],  # type: ignore[index]
                results["distances"][0],  # type: ignore[index]
            )
        ):
            chunk = Chunk(
                id=rid,
                document_id=meta.get("document_id", ""),
                content=doc,
                metadata=meta,
            )
            # cosine distance [0,2] → similarity [0,1]
            out.append(SearchResult(chunk=chunk, score=1.0 - dist / 2.0, rank=rank))

        return out

    async def delete_by_document(self, document_id: str) -> None:
        results = self._col.get(where={"document_id": document_id})
        if results["ids"]:
            self._col.delete(ids=results["ids"])
