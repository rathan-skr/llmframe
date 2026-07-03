from __future__ import annotations
import os
import tempfile
import shutil
import json
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Security, Depends, APIRouter
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ..core.config import FrameworkConfig
from ..core.types import Message, ChatResponse, RAGResponse
from ..core.logging import setup_logging, get_logger
from ..providers.anthropic_provider import AnthropicProvider
from ..rag.pipeline import RAGPipeline
from ..memory.conversation import ConversationStore

app = FastAPI(
    title="LLM Framework API",
    version="0.1.0",
    description="Production-grade RAG + multi-LLM REST API",
)

_config: FrameworkConfig | None = None
_llm: AnthropicProvider | None = None
_rag: RAGPipeline | None = None
_memory: ConversationStore = ConversationStore()
_log: logging.Logger = logging.getLogger("llmframe.api")

# ── Auth ─────────────────────────────────────────────────────────────────────

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _require_api_key(key: str | None = Security(_api_key_header)) -> None:
    secret = _get_config().api_secret_key
    if secret and key != secret:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# All routes mounted here require a valid X-API-Key header (when API_SECRET_KEY is set)
protected = APIRouter(dependencies=[Depends(_require_api_key)])


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup() -> None:
    cfg = _get_config()
    setup_logging(level=cfg.log.level, log_file=cfg.log.log_file)
    global _log
    _log = get_logger("api")
    auth_status = "enabled" if cfg.api_secret_key else "disabled"
    _log.info(
        f"LLM Framework API started | model={cfg.llm.model} "
        f"embedding={cfg.rag.embedding_provider} hybrid={cfg.rag.hybrid_search_enabled} "
        f"rerank={cfg.rag.reranker_enabled} auth={auth_status}"
    )


# ── Singletons ────────────────────────────────────────────────────────────────

def _get_config() -> FrameworkConfig:
    global _config
    if _config is None:
        _config = FrameworkConfig.from_env()
    return _config


def _get_llm() -> AnthropicProvider:
    global _llm
    if _llm is None:
        cfg = _get_config()
        _llm = AnthropicProvider(api_key=cfg.anthropic_api_key, config=cfg.llm)
    return _llm


def _get_rag() -> RAGPipeline:
    global _rag
    if _rag is None:
        _rag = RAGPipeline(llm=_get_llm(), config=_get_config())
    return _rag


# ── Request schemas ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    messages: list[Message]
    model: str | None = None
    stream: bool = False
    session_id: str | None = None


class IngestRequest(BaseModel):
    content: str
    source: str = ""
    metadata: dict = {}


class QueryRequest(BaseModel):
    question: str
    k: int = 5


# ── Public endpoints (no auth) ────────────────────────────────────────────────

@app.get("/health")
async def health():
    _log.debug("Health check")
    return {"status": "ok", "version": "0.1.0"}


# ── Protected endpoints ───────────────────────────────────────────────────────

@protected.post("/chat")
async def chat(request: ChatRequest):
    try:
        session_id = request.session_id or _memory.create_session()
        system_msgs = [m for m in request.messages if m.role == "system"]
        new_msgs = [m for m in request.messages if m.role != "system"]
        history = _memory.get(session_id)
        full_messages = system_msgs + history + new_msgs

        kwargs = {}
        if request.model:
            kwargs["model"] = request.model

        if request.stream:
            async def generate():
                full_text = ""
                async for chunk in _get_llm().stream(full_messages, **kwargs):
                    full_text += chunk
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                if new_msgs:
                    _memory.append(session_id, new_msgs[-1], Message(role="assistant", content=full_text))
                yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")

        response = await _get_llm().chat(full_messages, **kwargs)
        if new_msgs:
            _memory.append(session_id, new_msgs[-1], Message(role="assistant", content=response.content))
        response.session_id = session_id
        _log.info(f"Chat | session={session_id} model={response.model} tokens={response.input_tokens}in/{response.output_tokens}out")
        return response

    except Exception as e:
        _log.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@protected.get("/sessions/{session_id}")
async def get_session(session_id: str):
    history = _memory.get(session_id)
    if not history:
        raise HTTPException(status_code=404, detail="Session not found or empty")
    return {
        "session_id": session_id,
        "message_count": len(history),
        "messages": [{"role": m.role, "content": m.content} for m in history],
    }


@protected.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    deleted = _memory.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": session_id}


@protected.post("/ingest")
async def ingest(request: IngestRequest):
    try:
        _log.info(f"Ingest | source={request.source!r} len={len(request.content)}")
        doc = await _get_rag().ingest(
            request.content, source=request.source, metadata=request.metadata
        )
        return {"document_id": doc.id, "source": doc.source, "status": "stored"}
    except Exception as e:
        _log.error(f"Ingest error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@protected.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    suffix = "." + (file.filename or "upload").rsplit(".", 1)[-1]
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        doc = await _get_rag().ingest_file(tmp_path)
        return {"document_id": doc.id, "source": file.filename, "status": "stored"}
    except Exception as e:
        _log.error(f"Ingest file error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@protected.post("/query", response_model=RAGResponse)
async def query(request: QueryRequest):
    try:
        _log.info(f"Query | q={request.question[:80]!r} k={request.k}")
        return await _get_rag().query(request.question, k=request.k)
    except Exception as e:
        _log.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@protected.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    try:
        await _get_rag().delete_document(document_id)
        return {"deleted": document_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.include_router(protected)
