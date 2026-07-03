from __future__ import annotations
import time
import uuid
from ..core.types import Document, RAGResponse, Message
from ..core.config import FrameworkConfig
from ..core.logging import get_logger
from ..providers.base import BaseLLMProvider
from .chunking import TextChunker
from .embeddings import create_embedding_provider
from .vectorstore import BaseVectorStore, ChromaVectorStore


_SYSTEM = """You are a knowledgeable assistant. Answer using ONLY the provided context.
If the context does not contain enough information, say so directly.
Always indicate which source(s) your answer is drawn from."""

_TEMPLATE = """\
Context:
{context}

Question: {query}

Answer based solely on the context above:"""


class RAGPipeline:
    def __init__(
        self,
        llm: BaseLLMProvider,
        config: FrameworkConfig | None = None,
        vector_store: BaseVectorStore | None = None,
    ):
        self._llm = llm
        self._cfg = config or FrameworkConfig.from_env()
        self._log = get_logger("pipeline")

        self._chunker = TextChunker(
            chunk_size=self._cfg.rag.chunk_size,
            chunk_overlap=self._cfg.rag.chunk_overlap,
        )
        self._embedder = create_embedding_provider(
            provider=self._cfg.rag.embedding_provider,
            api_key=self._cfg.openai_api_key,
            model=self._cfg.rag.embedding_model,
        )
        self._store = vector_store or ChromaVectorStore(
            persist_dir=self._cfg.vector_store.chroma_persist_dir,
            collection=self._cfg.vector_store.collection_name,
        )

        # Optional: BM25 keyword index for hybrid search
        self._bm25 = None
        if self._cfg.rag.hybrid_search_enabled:
            from .hybrid_search import BM25Index
            self._bm25 = BM25Index()
            self._log.info("Hybrid search enabled (BM25 + vector → RRF)")

        # Optional: cross-encoder reranker
        self._reranker = None
        if self._cfg.rag.reranker_enabled:
            from .reranker import Reranker
            self._reranker = Reranker(model=self._cfg.rag.reranker_model)
            self._log.info(f"Reranker enabled: {self._cfg.rag.reranker_model}")

    async def ingest(
        self,
        content: str,
        source: str = "",
        metadata: dict | None = None,
    ) -> Document:
        t0 = time.perf_counter()
        doc = Document(id=str(uuid.uuid4()), content=content, source=source, metadata=metadata or {})
        chunks = self._chunker.chunk(doc)
        chunks = await self._embedder.embed_chunks(chunks)
        await self._store.add(chunks)

        if self._bm25:
            self._bm25.add(chunks)

        elapsed = time.perf_counter() - t0
        self._log.info(
            f"Ingest complete | doc={doc.id} source={source!r} chunks={len(chunks)} time={elapsed:.2f}s"
        )
        return doc

    async def ingest_file(self, file_path: str) -> Document:
        if file_path.endswith(".pdf"):
            content = self._read_pdf(file_path)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        return await self.ingest(content, source=file_path)

    async def query(self, question: str, k: int | None = None) -> RAGResponse:
        k = k or self._cfg.rag.max_retrieved_chunks

        # Fetch more candidates when reranking (reranker picks the best k from a wider pool)
        fetch_k = k * 3 if self._reranker else k

        # ── Step 1: Vector search ────────────────────────────────────────────
        t0 = time.perf_counter()
        embedding = await self._embedder.embed_query(question)
        vector_results = await self._store.search(embedding, k=fetch_k)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        # ── Step 2: Hybrid search (BM25 + RRF merge) ────────────────────────
        if self._bm25:
            from .hybrid_search import reciprocal_rank_fusion
            bm25_results = self._bm25.search(question, k=fetch_k)
            chunk_map = {r.chunk.id: r for r in vector_results}
            results = reciprocal_rank_fusion(
                vector_results, bm25_results, chunk_map, top_n=fetch_k
            )
            self._log.debug(f"Hybrid RRF merged {len(vector_results)} vector + {len(bm25_results)} BM25 → {len(results)} results")
        else:
            results = vector_results

        # ── Step 3: Rerank ───────────────────────────────────────────────────
        if self._reranker:
            results = self._reranker.rerank(question, results, top_k=k)
            self._log.debug(f"Reranked {fetch_k} candidates → top {k}")
        else:
            results = results[:k]

        self._log.info(
            f"Retrieval | chunks={len(results)} retrieval={retrieval_ms:.0f}ms "
            f"hybrid={self._bm25 is not None} rerank={self._reranker is not None}"
        )

        # ── Step 4: Build context + generate answer ──────────────────────────
        context = "\n\n---\n\n".join(
            f"[Source {i + 1}: {r.chunk.metadata.get('source', 'unknown')}]\n{r.chunk.content}"
            for i, r in enumerate(results)
        )

        messages = [
            Message(role="system", content=_SYSTEM),
            Message(role="user", content=_TEMPLATE.format(context=context, query=question)),
        ]

        t1 = time.perf_counter()
        response = await self._llm.chat(messages)
        gen_ms = (time.perf_counter() - t1) * 1000

        self._log.info(
            f"Generation | model={response.model} "
            f"tokens={response.input_tokens}in/{response.output_tokens}out "
            f"gen={gen_ms:.0f}ms"
        )

        return RAGResponse(
            answer=response.content,
            sources=results,
            model=response.model,
            provider=response.provider,
            query=question,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

    async def delete_document(self, document_id: str) -> None:
        await self._store.delete_by_document(document_id)
        if self._bm25:
            self._bm25.remove(document_id)
        self._log.info(f"Deleted | document_id={document_id}")

    def _read_pdf(self, path: str) -> str:
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
