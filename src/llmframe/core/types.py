from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatResponse(BaseModel):
    content: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    thinking: str | None = None
    session_id: str | None = None


class Document(BaseModel):
    id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str = ""


class Chunk(BaseModel):
    id: str
    document_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None


class SearchResult(BaseModel):
    chunk: Chunk
    score: float
    rank: int


class RAGResponse(BaseModel):
    answer: str
    sources: list[SearchResult]
    model: str
    provider: str
    query: str
    input_tokens: int = 0
    output_tokens: int = 0
