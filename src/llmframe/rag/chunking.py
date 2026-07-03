from __future__ import annotations
import uuid
from ..core.types import Document, Chunk


class TextChunker:
    """Recursive character-based text splitter — tries paragraph → sentence → word boundaries."""

    _SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> list[Chunk]:
        texts = self._split(document.content)
        return [
            Chunk(
                id=str(uuid.uuid4()),
                document_id=document.id,
                content=text,
                metadata={
                    **document.metadata,
                    "source": document.source,
                    "chunk_index": i,
                    "document_id": document.id,
                },
            )
            for i, text in enumerate(texts)
        ]

    def _split(self, text: str) -> list[str]:
        for sep in self._SEPARATORS:
            if sep == "" or sep in text:
                splits = text.split(sep) if sep else list(text)
                return self._merge(splits, sep)
        return [text]

    def _merge(self, splits: list[str], sep: str) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for s in splits:
            if not s:
                continue
            s_len = len(s)
            sep_len = len(sep) if current else 0

            if current_len + sep_len + s_len > self.chunk_size and current:
                chunks.append(sep.join(current))
                # trim from front until we're within overlap budget
                while current and current_len > self.chunk_overlap:
                    removed = current.pop(0)
                    current_len -= len(removed) + len(sep)

            current.append(s)
            current_len += s_len + (len(sep) if len(current) > 1 else 0)

        if current:
            chunks.append(sep.join(current))

        return [c for c in chunks if c.strip()]
