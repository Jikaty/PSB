from __future__ import annotations

from dataclasses import dataclass

from kb_loader import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    locator: str
    text: str
    notes: str
    start: int
    end: int


def chunk_company_document(doc: Document) -> list[Chunk]:
    raise NotImplementedError


def chunk_law(doc: Document) -> list[Chunk]:
    raise NotImplementedError


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    raise NotImplementedError
