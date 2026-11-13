from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    doc_id: str
    file_name: str
    title: str
    doc_type: str
    number: str
    date: str
    parties: str
    status: str
    valid_from: str
    valid_to: str
    relates_to: list[str] = field(default_factory=list)
    text: str = ""
    suspicious: list[tuple[int, int]] = field(default_factory=list)


def find_archive(data_dir: Path) -> Path:
    raise NotImplementedError


def read_documents(archive: Path) -> list[tuple[str, str]]:
    raise NotImplementedError


def build_card(file_name: str, text: str) -> Document:
    raise NotImplementedError


def mark_suspicious(text: str) -> list[tuple[int, int]]:
    raise NotImplementedError


def load_knowledge_base(data_dir: Path) -> list[Document]:
    raise NotImplementedError
