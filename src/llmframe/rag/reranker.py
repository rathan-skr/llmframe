from __future__ import annotations
from ..core.types import SearchResult


class Reranker:
    """
    Cross-encoder reranker — scores every (query, chunk) pair directly.
    More accurate than vector similarity but slower (runs all pairs through a model).
    Used AFTER vector retrieval: fetch k*3 candidates, rerank, return top k.
    """

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model)
        self._model_name = model

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not results:
            return results

        pairs  = [(query, r.chunk.content) for r in results]
        scores = self._model.predict(pairs)

        ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            SearchResult(chunk=r.chunk, score=float(s), rank=i)
            for i, (r, s) in enumerate(ranked)
        ]
