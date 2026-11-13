from __future__ import annotations

from dataclasses import dataclass, field

from chunker import Chunk
from kb_loader import Document


@dataclass
class Retrieval:
    selected_docs: list[Document] = field(default_factory=list)
    law_chunks: list[Chunk] = field(default_factory=list)
    flags: dict[str, bool] = field(default_factory=dict)


class Retriever:
    def __init__(self, docs: list[Document], chunks: list[Chunk]) -> None:
        raise NotImplementedError

    def search(self, query: str, history_user_turns: list[str]) -> Retrieval:
        raise NotImplementedError
