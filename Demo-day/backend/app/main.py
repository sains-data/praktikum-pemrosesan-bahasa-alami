"""
FastAPI backend — Demo-day chatbot (BERT / Transformer).
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.app import config
from backend.app.engines.factory import build_engine

_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    _engine = build_engine()
    _engine.load()
    yield
    _engine = None


app = FastAPI(
    title="Demo-day Chatbot API",
    description="Helpdesk chatbot — backend untuk bot Discord (Itera NLP)",
    version="1.0.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=512, description="Pesan pengguna")
    beam_size: int = Field(3, ge=1, le=10, description="Lebar beam search (BERT)")


class ChatResponse(BaseModel):
    reply: str
    model_type: str


@app.get("/")
def root():
    return {
        "service": "demo-day-chatbot",
        "model_type": config.MODEL_TYPE,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    ok = _engine is not None
    return {
        "status": "ok" if ok else "unavailable",
        "model_type": config.MODEL_TYPE,
        "model_loaded": ok,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model belum siap.")
    try:
        reply = _engine.reply(body.message, beam_size=body.beam_size)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ChatResponse(reply=reply, model_type=_engine.name)
