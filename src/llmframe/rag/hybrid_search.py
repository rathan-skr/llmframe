from __future__ import annotations
from ..core.types import Chunk, SearchResult


class BM25Index:
    """
    In-memory BM25 keyword index — the "keyword" half of hybrid search.
    Keeps a copy of every ingested chunk for scoring against exact/partial word matches.
    Lost on restart (acceptable since ChromaDB is the source of truth).
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._bm25 = None

    def add(self, chunks: list[Chunk]) -> None:
        self._chunks.extend(chunks)
        self._rebuild()

    def remove(self, document_id: str) -> None:
        self._chunks = [c for c in self._chunks if c.metadata.get("document_id") != document_id]
        self._rebuild()

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Returns list of (chunk_id, bm25_score) sorted by score descending."""
        if not self._bm25 or not self._chunks:
            return []
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        return [(self._chunks[i].id, float(s)) for i, s in indexed if s > 0]

    def _rebuild(self) -> None:
        if not self._chunks:
            self._bm25 = None
            return
        from rank_bm25 import BM25Okapi
        corpus = [c.content.lower().split() for c in self._chunks]
        self._bm25 = BM25Okapi(corpus)


def reciprocal_rank_fusion(
    vector_results: list[SearchResult],
    bm25_results: list[tuple[str, float]],
    chunk_map: dict[str, SearchResult],
    rrf_k: int = 60,
    top_n: int = 5,
) -> list[SearchResult]:
    """
    Merges vector search + BM25 results using Reciprocal Rank Fusion.
    RRF score = sum of 1/(k + rank) across both ranked lists.
    Higher score = appeared near top in more lists = more relevant.
    """
    scores: dict[str, float] = {}

    for rank, result in enumerate(vector_results):
        cid = result.chunk.id
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)

    for rank, (chunk_id, _) in enumerate(bm25_results):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_n]

    out = []
    for rank, cid in enumerate(sorted_ids):
        if cid in chunk_map:
            r = chunk_map[cid]
            out.append(SearchResult(chunk=r.chunk, score=scores[cid], rank=rank))
    return out
