from __future__ import annotations

import httpx

from config import Settings


def make_client() -> httpx.Client:
    raise NotImplementedError


def chat(client: httpx.Client, settings: Settings, messages: list[dict], json_mode: bool = True) -> str:
    raise NotImplementedError


def embed(client: httpx.Client, settings: Settings, texts: list[str]) -> list[list[float]]:
    raise NotImplementedError


def rerank(client: httpx.Client, settings: Settings, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
    raise NotImplementedError
