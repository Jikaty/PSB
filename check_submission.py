from __future__ import annotations

import argparse
import csv
import io
import sys
from collections import Counter
from pathlib import Path

from config import SETTINGS

MAX_CHARS = 10000
SOFT_WORDS = 150


def read_text(path: Path, problems: list[str]) -> str | None:
    if not path.is_file():
        problems.append(f"нет файла {path}")
        return None
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        problems.append(f"{path.name}: есть BOM")
        raw = raw[3:]
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        problems.append(f"{path.name}: не UTF-8 ({exc.reason}, байт {exc.start})")
        return None


def question_ids(path: Path, problems: list[str]) -> list[str] | None:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = {(name or "").strip().lower(): name for name in reader.fieldnames or []}
            if "id" not in fields:
                problems.append(f"{path.name}: нет колонки id")
                return None
            return [str(row[fields["id"]] or "").strip() for row in reader]
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        problems.append(f"{path.name}: не читается ({type(exc).__name__})")
        return None


def check_rows(rows: list[list[str]], problems: list[str], notes: list[str]) -> list[str]:
    if not rows or rows[0] != ["id", "answer"]:
        problems.append(f"заголовок {rows[0] if rows else 'отсутствует'}, нужен ['id', 'answer']")
    ids = []
    long_answers = 0
    for number, row in enumerate(rows[1:], start=2):
        if len(row) != 2:
            problems.append(f"строка {number}: {len(row)} полей вместо 2")
            continue
        qid, answer = row
        ids.append(qid)
        if not answer.strip():
            problems.append(f"строка {number} (id {qid}): пустой ответ")
        if len(answer) > MAX_CHARS:
            problems.append(f"строка {number} (id {qid}): {len(answer)} символов > {MAX_CHARS}")
        if "\n" in answer or "\r" in answer:
            problems.append(f"строка {number} (id {qid}): перевод строки в ответе")
        long_answers += len(answer.split()) > SOFT_WORDS
    if long_answers:
        notes.append(f"ответов длиннее {SOFT_WORDS} слов: {long_answers} (рекомендация, не ошибка)")
    return ids


def compare_ids(ids: list[str], expected: list[str], problems: list[str]) -> None:
    duplicates = sorted(qid for qid, number in Counter(ids).items() if number > 1)
    if duplicates:
        problems.append("повторы id: " + ", ".join(duplicates[:20]))
    missing = [qid for qid in dict.fromkeys(expected) if qid not in set(ids)]
    extra = [qid for qid in dict.fromkeys(ids) if qid not in set(expected)]
    if missing:
        problems.append(f"нет id ({len(missing)}): " + ", ".join(missing[:20]))
    if extra:
        problems.append(f"лишние id ({len(extra)}): " + ", ".join(extra[:20]))
    if not missing and not extra and not duplicates and ids != list(dict.fromkeys(expected)):
        problems.append("порядок id отличается от questions.csv")


def check(submission: Path, questions: Path) -> tuple[list[str], list[str]]:
    problems: list[str] = []
    notes: list[str] = []
    text = read_text(submission, problems)
    expected = question_ids(questions, problems)
    if text is None:
        return problems, notes
    if "\r" in text:
        notes.append("в файле есть символ CR (ожидались окончания строк LF)")
    try:
        rows = list(csv.reader(io.StringIO(text, newline="")))
    except csv.Error as exc:
        problems.append(f"CSV не разбирается: {exc}")
        return problems, notes
    ids = check_rows(rows, problems, notes)
    if expected is not None:
        compare_ids(ids, expected, problems)
        notes.append(f"строк: {len(ids)}, вопросов: {len(expected)}")
    return problems, notes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", nargs="?", default="submission.csv")
    parser.add_argument("questions", nargs="?", default=str(SETTINGS.data_dir / "questions.csv"))
    args = parser.parse_args()
    problems, notes = check(Path(args.submission), Path(args.questions))
    for note in notes:
        print("  " + note)
    if problems:
        print("Нарушения:")
        for problem in problems:
            print("- " + problem)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
