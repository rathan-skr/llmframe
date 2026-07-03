from __future__ import annotations
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    provider: str = "anthropic"
    model: str = "claude-opus-4-8"
    max_tokens: int = 8096
    temperature: float = 1.0
    use_thinking: bool = True
    stream: bool = True


@dataclass
class RAGConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_retrieved_chunks: int = 5
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    hybrid_search_enabled: bool = False


@dataclass
class LogConfig:
    level: str = "INFO"
    log_file: str = "./logs/llmframe.log"


@dataclass
class VectorStoreConfig:
    store_type: str = "chroma"
    chroma_persist_dir: str = "./data/chroma"
    collection_name: str = "documents"
    pinecone_api_key: str = ""
    pinecone_index: str = ""


@dataclass
class FrameworkConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    log: LogConfig = field(default_factory=LogConfig)
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    api_secret_key: str = field(default_factory=lambda: os.getenv("API_SECRET_KEY", ""))

    @classmethod
    def from_env(cls) -> FrameworkConfig:
        cfg = cls()
        cfg.api_secret_key = os.getenv("API_SECRET_KEY", "")
        cfg.llm.provider = os.getenv("DEFAULT_LLM_PROVIDER", "anthropic")
        cfg.llm.model = os.getenv("DEFAULT_MODEL", "claude-opus-4-8")
        cfg.rag.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        cfg.rag.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        cfg.rag.max_retrieved_chunks = int(os.getenv("MAX_RETRIEVED_CHUNKS", "5"))
        cfg.rag.embedding_provider = os.getenv("EMBEDDING_PROVIDER", "openai")
        cfg.rag.embedding_model = os.getenv("EMBEDDING_MODEL", "")
        cfg.rag.reranker_enabled = os.getenv("RERANKER_ENABLED", "false").lower() == "true"
        cfg.rag.reranker_model = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        cfg.rag.hybrid_search_enabled = os.getenv("HYBRID_SEARCH_ENABLED", "false").lower() == "true"
        cfg.log.level = os.getenv("LOG_LEVEL", "INFO")
        cfg.log.log_file = os.getenv("LOG_FILE", "./logs/llmframe.log")
        cfg.vector_store.store_type = os.getenv("VECTOR_STORE_TYPE", "chroma")
        cfg.vector_store.chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
        cfg.vector_store.collection_name = os.getenv("VECTOR_COLLECTION", "documents")
        cfg.vector_store.pinecone_api_key = os.getenv("PINECONE_API_KEY", "")
        cfg.vector_store.pinecone_index = os.getenv("PINECONE_INDEX", "")
        return cfg
