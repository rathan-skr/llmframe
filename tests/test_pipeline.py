import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.llmframe.core.types import ChatResponse, SearchResult, Chunk
from src.llmframe.core.config import FrameworkConfig
from src.llmframe.rag.pipeline import RAGPipeline


def _fake_config() -> FrameworkConfig:
    cfg = FrameworkConfig(anthropic_api_key="test", openai_api_key="test")
    return cfg


def _fake_search_result(content: str = "Relevant text.", source: str = "doc.txt") -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            id="chunk-1",
            document_id="doc-1",
            content=content,
            metadata={"source": source, "document_id": "doc-1"},
        ),
        score=0.92,
        rank=0,
    )


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=ChatResponse(
        content="The answer is 42.",
        model="claude-opus-4-8",
        provider="anthropic",
        input_tokens=150,
        output_tokens=25,
    ))
    return llm


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.add = AsyncMock()
    store.search = AsyncMock(return_value=[_fake_search_result()])
    store.delete_by_document = AsyncMock()
    return store


@pytest.fixture
def pipeline(mock_llm, mock_store):
    """RAGPipeline with all external I/O mocked out."""
    async def _embed_chunks(chunks):
        return [c.model_copy(update={"embedding": [0.1] * 8}) for c in chunks]

    with patch("src.llmframe.rag.pipeline.create_embedding_provider") as mock_factory:
        embedder = MagicMock()
        embedder.embed_chunks = AsyncMock(side_effect=_embed_chunks)
        embedder.embed_query = AsyncMock(return_value=[0.1] * 8)
        mock_factory.return_value = embedder

        p = RAGPipeline(llm=mock_llm, config=_fake_config(), vector_store=mock_store)
        yield p


@pytest.mark.asyncio
async def test_ingest_returns_document_with_id(pipeline, mock_store):
    doc = await pipeline.ingest("Some content about AI.", source="ai.txt")
    assert doc.id  # has a UUID
    assert doc.source == "ai.txt"
    mock_store.add.assert_called_once()


@pytest.mark.asyncio
async def test_ingest_calls_embedder_and_store(pipeline, mock_store):
    await pipeline.ingest("Text content", source="notes.txt")
    # Embedder was called (chunks had embeddings set before store.add)
    call_args = mock_store.add.call_args[0][0]  # list of chunks
    assert all(c.embedding is not None for c in call_args)


@pytest.mark.asyncio
async def test_query_returns_rag_response(pipeline):
    result = await pipeline.query("What is the answer?")
    assert result.answer == "The answer is 42."
    assert result.query == "What is the answer?"
    assert result.model == "claude-opus-4-8"
    assert len(result.sources) == 1


@pytest.mark.asyncio
async def test_query_injects_retrieved_context_into_prompt(pipeline, mock_llm):
    await pipeline.query("What is the answer?")
    messages_sent = mock_llm.chat.call_args[0][0]
    user_msg = next(m for m in messages_sent if m.role == "user")
    assert "Relevant text." in user_msg.content


@pytest.mark.asyncio
async def test_query_passes_source_in_context(pipeline, mock_llm):
    await pipeline.query("test question")
    messages_sent = mock_llm.chat.call_args[0][0]
    user_msg = next(m for m in messages_sent if m.role == "user")
    assert "doc.txt" in user_msg.content


@pytest.mark.asyncio
async def test_query_includes_system_prompt(pipeline, mock_llm):
    await pipeline.query("test")
    messages_sent = mock_llm.chat.call_args[0][0]
    system_msgs = [m for m in messages_sent if m.role == "system"]
    assert len(system_msgs) == 1
    assert "context" in system_msgs[0].content.lower()


@pytest.mark.asyncio
async def test_delete_document_calls_store(pipeline, mock_store):
    await pipeline.delete_document("doc-abc")
    mock_store.delete_by_document.assert_called_once_with("doc-abc")


@pytest.mark.asyncio
async def test_empty_store_returns_empty_sources(pipeline, mock_store):
    mock_store.search = AsyncMock(return_value=[])
    result = await pipeline.query("anything")
    assert result.sources == []
    assert result.answer == "The answer is 42."  # LLM still called (with empty context)
