from __future__ import annotations

SYSTEM_PROMPT = ""


def build_system_prompt(today: str) -> str:
    raise NotImplementedError


def build_user_prompt(catalog: str, context: str, history: str, question: str, event_date: str) -> str:
    raise NotImplementedError
