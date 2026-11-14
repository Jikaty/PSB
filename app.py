from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import pipeline
from config import SETTINGS, env, env_int
from kb_loader import Document, load_kb_dir

ROOT = Path(__file__).resolve().parent
KB_DEFAULT = "data/knowledge-base"
CACHE_DIR = ROOT / "data" / "cache"
STATIC_DIR = ROOT / "static"
MAX_QUESTION = 20000


class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION)
    as_of_date: date | None = None


class Source(BaseModel):
    n: int
    doc_id: str
    title: str
    status: str
    locator: str
    start: int
    end: int
    text: str


class Timings(BaseModel):
    retrieval: int
    llm: int
    total: int


class Answer(BaseModel):
    answer: str
    sources: list[Source]
    warnings: list[str]
    timings_ms: Timings


class Card(BaseModel):
    doc_id: str
    title: str
    type: str
    status: str
    valid_from: str | None
    valid_to: str | None


class Span(BaseModel):
    start: int
    end: int


class FullDocument(Card):
    text: str
    suspicious: list[Span]


def kb_dir() -> Path:
    path = Path(env("KB_DIR", default=KB_DEFAULT))
    return path if path.is_absolute() or path.exists() else ROOT / path


def start() -> None:
    try:
        docs = load_kb_dir(kb_dir(), SETTINGS.data_dir)
    except FileNotFoundError:
        raise RuntimeError(f"База знаний не найдена: KB_DIR={kb_dir()}") from None
    engine = pipeline.start_engine(docs, CACHE_DIR)
    state = "выкл" if engine.retriever.embeddings_off else "вкл"
    print(f"База знаний: документов {len(docs)}, фрагментов {len(engine.chunks)}, эмбеддинги {state}", file=sys.stderr)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await run_in_threadpool(start)
    yield


app = FastAPI(title="Юридический ассистент", lifespan=lifespan)


def card(doc: Document) -> dict:
    return {
        "doc_id": doc.doc_id, "title": doc.title, "type": doc.doc_type, "status": doc.status,
        "valid_from": doc.valid_from or None, "valid_to": doc.valid_to or None,
    }


@app.post("/api/ask", response_model=Answer)
def ask(body: Ask) -> dict:
    if not body.question.strip():
        raise HTTPException(status_code=422, detail="Пустой вопрос")
    return pipeline.answer_with_sources(body.question, body.as_of_date, cite=True)


@app.get("/api/docs", response_model=list[Card])
def list_docs() -> list[dict]:
    return [card(doc) for doc in pipeline.get_engine().docs]


@app.get("/api/docs/{doc_id}", response_model=FullDocument)
def read_doc(doc_id: str) -> dict:
    doc = pipeline.get_engine().retriever.by_id.get(doc_id.lower())
    if doc is None:
        raise HTTPException(status_code=404, detail="Документ не найден")
    return {**card(doc), "text": doc.text, "suspicious": [{"start": start, "end": end} for start, end in doc.suspicious]}


@app.get("/health")
def health() -> dict:
    engine = pipeline.get_engine()
    return {
        "status": "ok", "docs": len(engine.docs), "chunks": len(engine.chunks),
        "embeddings": not engine.retriever.embeddings_off,
    }


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=env("HOST", default="0.0.0.0"), port=env_int("PORT", 8000, 1, 65535))
