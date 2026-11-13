from __future__ import annotations


def split_dialog(raw_question: str) -> tuple[list[tuple[str, str]], str]:
    raise NotImplementedError


def answer_question(raw_question: str) -> dict:
    raise NotImplementedError
