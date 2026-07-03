import pytest
from src.llmframe.core.types import Document
from src.llmframe.rag.chunking import TextChunker


def _doc(content: str, source: str = "test.txt") -> Document:
    return Document(id="test-doc", content=content, source=source)


def test_short_text_is_single_chunk():
    chunker = TextChunker(chunk_size=1000, chunk_overlap=100)
    chunks = chunker.chunk(_doc("Hello world."))
    assert len(chunks) == 1
    assert chunks[0].content == "Hello world."


def test_long_text_splits_into_multiple_chunks():
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk(_doc("word " * 100))  # 500 chars
    assert len(chunks) > 1


def test_chunk_inherits_document_id():
    chunker = TextChunker()
    doc = Document(id="doc-999", content="Some text.", source="file.txt")
    chunks = chunker.chunk(doc)
    assert all(c.document_id == "doc-999" for c in chunks)


def test_chunk_metadata_includes_source_and_index():
    chunker = TextChunker()
    doc = Document(id="d1", content="Hello world.", source="notes.txt", metadata={"author": "Alice"})
    chunks = chunker.chunk(doc)
    assert chunks[0].metadata["source"] == "notes.txt"
    assert chunks[0].metadata["author"] == "Alice"
    assert chunks[0].metadata["chunk_index"] == 0


def test_empty_document_returns_no_chunks():
    chunker = TextChunker()
    chunks = chunker.chunk(_doc(""))
    assert chunks == []


def test_whitespace_only_returns_no_chunks():
    chunker = TextChunker()
    chunks = chunker.chunk(_doc("   \n\n   "))
    assert chunks == []


def test_no_chunk_exceeds_size_limit():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    # Each paragraph is ~22 chars so the merger groups 3-4 per chunk, never exceeding 100
    paragraphs = [f"Para {i} text here." for i in range(30)]
    content = "\n\n".join(paragraphs)
    chunks = chunker.chunk(_doc(content))
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.content) <= 100, f"Chunk too long: {len(c.content)}"


def test_chunk_ids_are_unique():
    chunker = TextChunker(chunk_size=50, chunk_overlap=5)
    chunks = chunker.chunk(_doc("word " * 200))
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique"


def test_sequential_chunk_indices():
    chunker = TextChunker(chunk_size=50, chunk_overlap=5)
    chunks = chunker.chunk(_doc("word " * 200))
    for i, c in enumerate(chunks):
        assert c.metadata["chunk_index"] == i
