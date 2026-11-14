from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta

from chunker import Chunk
from kb_loader import DATE_ANY, DATE_NUM, DATE_WORDS, STATUS_VOID, VOID_BANNER, Document, format_dmy, to_iso
from retriever import DOC_NUMBER, MONTH_KEYS, MONTH_YEAR, YEAR

MAX_PERIODS = 3
MAX_CLAUSES = 2
MAX_LAW_NOTES = 6
BEFORE = re.compile(r"\b(?:до|ранее|раньше|по состоянию до)\s*$", re.I)
CLAUSE = re.compile(r"(?:^|[\s«(])(с|начиная с|после)\s+(" + DATE_ANY + ")", re.I)
LAW_CHANGE = re.compile(
    r"(измен[её]н\w*|дополнен\w*|утратил\w*\s+силу|в\s+новой\s+редакции|внесены\s+изменения)[^\n]*?\bс\s+(" + DATE_ANY + ")",
    re.I,
)
MONTH_NAMES = (
    "январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь",
)
IN_FORCE = "ДЕЙСТВОВАЛ"
NOT_YET = "ЕЩЁ НЕ ДЕЙСТВОВАЛ"
EXPIRED = "УЖЕ НЕ ДЕЙСТВОВАЛ"
PARTLY = "ДЕЙСТВОВАЛ ЧАСТЬ ПЕРИОДА"


@dataclass(frozen=True)
class Period:
    start: date | None
    end: date
    label: str


@dataclass(frozen=True)
class Clause:
    start: date
    text: str


def dmy(value: date) -> str:
    return value.strftime("%d.%m.%Y")


def parse_iso(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def make_period(first: date, last: date, label: str, before: bool) -> Period:
    if before:
        return Period(None, first - timedelta(days=1), "до " + dmy(first))
    return Period(first, last, label)


def find_periods(text: str) -> list[Period]:
    rest = DOC_NUMBER.sub(lambda m: " " * len(m.group()), text)
    found: list[tuple[int, Period]] = []

    def take(pattern: re.Pattern, build) -> None:
        nonlocal rest
        for match in pattern.finditer(rest):
            span = build(match)
            if span:
                found.append((match.start(), make_period(*span, bool(BEFORE.search(rest[:match.start()])))))
        rest = pattern.sub(lambda m: " " * len(m.group()), rest)

    def day(match: re.Match):
        value = parse_iso(to_iso(match.group()))
        return (value, value, dmy(value)) if value else None

    def month(match: re.Match):
        number = next(i for i, key in enumerate(MONTH_KEYS, start=1) if match.group(1).lower().startswith(key))
        year = int(match.group(2))
        last = calendar.monthrange(year, number)[1]
        return date(year, number, 1), date(year, number, last), f"{MONTH_NAMES[number - 1]} {year} г."

    def year(match: re.Match):
        value = int(match.group())
        return date(value, 1, 1), date(value, 12, 31), f"{value} год"

    take(DATE_NUM, day)
    take(DATE_WORDS, day)
    take(MONTH_YEAR, month)
    take(YEAR, year)
    result: list[Period] = []
    for _, period in sorted(found, key=lambda item: item[0]):
        if period not in result:
            result.append(period)
    return result[:MAX_PERIODS]


def doc_status(doc: Document, period: Period) -> str:
    start, end = parse_iso(doc.valid_from), parse_iso(doc.valid_to)
    if start and period.end < start:
        return f"{NOT_YET} (действует с {dmy(start)})"
    if end and period.start and period.start > end:
        return f"{EXPIRED} (действовал по {dmy(end)})"
    partial = (start and (period.start is None or period.start < start)) or (end and period.end > end)
    span = " ".join(part for part in ("с " + dmy(start) if start else "", "по " + dmy(end) if end else "") if part)
    if partial:
        return f"{PARTLY} ({span})"
    return f"{IN_FORCE} ({span})" if span else IN_FORCE


def find_clauses(doc: Document, visible: str) -> list[Clause]:
    base = parse_iso(doc.valid_from) or parse_iso(doc.date)
    clauses: list[Clause] = []
    for line in visible.split("\n"):
        if VOID_BANNER.search(line) or "утрачива" in line.lower():
            continue
        for match in CLAUSE.finditer(line):
            value = parse_iso(to_iso(match.group(2)))
            if value is None or (base and value <= base):
                continue
            first = value + timedelta(days=1) if match.group(1).lower() == "после" else value
            clause = Clause(first, " ".join(line.split())[:300])
            if clause not in clauses:
                clauses.append(clause)
    return clauses[:MAX_CLAUSES]


def clause_status(clause: Clause, period: Period) -> str:
    if period.end < clause.start:
        return f"ЕЩЁ НЕ ПРИМЕНЯЛОСЬ (применяется с {dmy(clause.start)})"
    if period.start is None or period.start < clause.start:
        return f"ПРИМЕНЯЛОСЬ ЧАСТЬ ПЕРИОДА (с {dmy(clause.start)})"
    return f"ПРИМЕНЯЛОСЬ (с {dmy(clause.start)})"


def catalog_marks(docs: list[Document], periods: list[Period]) -> dict[str, str]:
    marks = {}
    for doc in docs:
        if doc.is_law or not (doc.valid_from or doc.valid_to):
            continue
        marks[doc.doc_id] = "; ".join(f"на дату «{period.label}»: {doc_status(doc, period)}" for period in periods)
    return {doc_id: mark for doc_id, mark in marks.items() if mark}


def clause_lines(docs: list[Document], texts: dict[str, str], periods: list[Period]) -> list[str]:
    lines = []
    for doc in docs:
        if doc.is_law:
            continue
        for clause in find_clauses(doc, texts.get(doc.doc_id, "")):
            for period in periods:
                lines.append(f"{doc.title}: условие «{clause.text}» на дату «{period.label}» {clause_status(clause, period)}")
    return lines


def law_lines(docs: list[Document], chunks: list[Chunk], periods: list[Period]) -> list[str]:
    by_id = {doc.doc_id: doc for doc in docs}
    earliest = min((period.start or date.min for period in periods), default=None)
    if earliest is None:
        return []
    lines: list[tuple[date, str]] = []
    exported: set[str] = set()
    for chunk in chunks:
        doc = by_id.get(chunk.doc_id)
        if doc is None or not doc.is_law:
            continue
        exported.add(doc.exported)
        for match in LAW_CHANGE.finditer(chunk.notes or ""):
            changed = parse_iso(to_iso(match.group(2)))
            if changed and changed > earliest:
                line = f"{doc.law}, {chunk.locator}: норма изменена с {dmy(changed)}; редакции, действовавшей до этой даты, в базе нет"
                if all(line != other for _, other in lines):
                    lines.append((changed, line))
    result = [line for _, line in sorted(lines, key=lambda item: item[0], reverse=True)[:MAX_LAW_NOTES]]
    stamp = next((format_dmy(value) if "-" in value else value for value in sorted(exported) if value), "")
    if exported and any((period.start or date.min) < (parse_iso(to_iso(stamp)) or date.max) for period in periods):
        result.append(
            "Статьи законов есть в базе только в действующей редакции"
            + (f" (выгрузка {stamp})" if stamp else "")
            + "; прежних редакций и значений за прошлые периоды в базе нет"
        )
    return result


def contract_lines(bound: list[Document], docs: list[Document]) -> list[str]:
    void = {doc.doc_id for doc in docs if doc.status == STATUS_VOID}
    editions = [doc for doc in docs if not doc.is_law and (doc.doc_id in void or void & set(doc.relates_to))]
    lines = []
    for doc in bound:
        signed = parse_iso(doc.date)
        if doc.is_law or signed is None:
            continue
        period = Period(signed, signed, dmy(signed))
        others = [other for other in editions if other.doc_id != doc.doc_id and other.doc_id not in doc.relates_to]
        if others:
            states = "; ".join(f"{other.title} — {doc_status(other, period)}" for other in others)
            lines.append(f"{doc.title} заключён {dmy(signed)}; на эту дату: {states}")
    return lines


MAX_COMPARISONS = 8
REGULATION = "регламент"
AMOUNT = re.compile(
    r"(?<![\w.,/-])(\d{1,3}(?:[   ]\d{3})+|\d+)(?:[.,](\d+))?\s*"
    r"(млрд\w*|миллиард\w*|млн\w*|миллион\w*|тыс\w*)?\.?\s*(руб\w*|₽|р\.)?",
    re.I,
)
LIMIT_WORDS = re.compile(r"(?:свыше|более|выше|менее|ниже|превыша\w*|порог\w*|лимит\w*|от|до)\s*$", re.I)
THRESHOLD = re.compile(
    r"(не\s+более|не\s+менее|не\s+превышающ\w*|не\s+превыша\w*|превышающ\w*|свыше|более|менее|от|до)\s+"
    r"(\d{1,3}(?:[   ]\d{3})+|\d+)(?:[.,](\d+))?\s*(?:\([^)]*\)\s*)?"
    r"(млрд\w*|миллиард\w*|млн\w*|миллион\w*|тыс\w*)?\.?\s*(?:\([^)]*\)\s*)?руб",
    re.I,
)
POINT_START = re.compile(r"^\s*(\d+(?:\.\d+)+)\.?\s+")
SCALE_STEMS = (("млрд", 1e9), ("миллиард", 1e9), ("млн", 1e6), ("миллион", 1e6), ("тыс", 1e3))


@dataclass(frozen=True)
class Threshold:
    point: str
    op: str
    word: str
    value: float
    text: str


def scale_value(word: str | None) -> float:
    low = (word or "").lower()
    return next((value for stem, value in SCALE_STEMS if low.startswith(stem)), 1.0)


def number_value(whole: str, fraction: str | None, scale: str | None) -> float:
    digits = re.sub(r"\D", "", whole)
    value = float(digits + "." + fraction) if fraction else float(digits)
    return round(value * scale_value(scale), 2)


def money(value: float) -> str:
    whole = f"{int(value):,}".replace(",", " ")
    rest = round(value - int(value), 2)
    return (whole + f"{rest:.2f}"[1:].replace(".", ",") if rest else whole) + " руб."


def find_amounts(texts: list[str]) -> list[float]:
    amounts: list[float] = []
    for text in texts:
        rest = DOC_NUMBER.sub(lambda m: " " * len(m.group()), text)
        rest = re.sub(DATE_ANY, lambda m: " " * len(m.group()), rest, flags=re.I)
        for match in AMOUNT.finditer(rest):
            if not (match.group(3) or match.group(4)) or LIMIT_WORDS.search(rest[:match.start()]):
                continue
            value = number_value(match.group(1), match.group(2), match.group(3))
            if value >= 1 and value not in amounts:
                amounts.append(value)
    return amounts


def operator(word: str) -> str:
    low = " ".join(word.lower().split())
    if low in ("не более", "до") or low.startswith("не превыша"):
        return "≤"
    if low in ("не менее", "от"):
        return "≥"
    return "<" if low == "менее" else ">"


def holds(amount: float, op: str, limit: float) -> bool:
    return {">": amount > limit, "≥": amount >= limit, "<": amount < limit, "≤": amount <= limit}[op]


def find_thresholds(text: str) -> list[Threshold]:
    result: list[Threshold] = []
    for line in text.split("\n"):
        start = POINT_START.match(line)
        if not start or VOID_BANNER.search(line):
            continue
        sentence = " ".join(line[start.end():].split())
        for match in THRESHOLD.finditer(sentence):
            value = number_value(match.group(2), match.group(3), match.group(4))
            result.append(Threshold(start.group(1), operator(match.group(1)), " ".join(match.group(1).lower().split()), value, sentence[:300]))
    return result


def regulations(docs: list[Document]) -> list[Document]:
    return [doc for doc in docs if not doc.is_law and doc.doc_type.lower().startswith(REGULATION)]


def amount_periods(periods: list[Period], bound: list[Document], today: date) -> list[Period]:
    if periods:
        return periods
    dated = []
    for doc in bound:
        signed = parse_iso(doc.date)
        if signed and not doc.is_law and not doc.doc_type.lower().startswith(REGULATION):
            dated.append(Period(signed, signed, f"{dmy(signed)} (дата заключения: {doc.title})"))
    return dated or [Period(today, today, f"{dmy(today)} (сегодня)")]


def threshold_lines(
    texts: list[str], docs: list[Document], visible: dict[str, str], periods: list[Period], bound: list[Document], today: date,
) -> list[str]:
    amounts = find_amounts(texts)
    editions = regulations(docs)
    if not amounts or not editions:
        return []
    lines: list[str] = []
    for period in amount_periods(periods, bound, today):
        for doc in editions:
            state = doc_status(doc, period)
            if not state.startswith(IN_FORCE):
                continue
            part = ", действовала только часть периода" if state.startswith(PARTLY) else ""
            for limit in find_thresholds(visible.get(doc.doc_id, doc.text)):
                for amount in amounts:
                    sign = ">" if amount > limit.value else "=" if amount == limit.value else "<"
                    verdict = "требуется" if holds(amount, limit.op, limit.value) else "не требуется"
                    line = (
                        f"сумма {money(amount)} {sign} порог {money(limit.value)} (условие пункта: «{limit.word} {money(limit.value)}»); "
                        f"п. {limit.point}, {doc.title} — редакция, применимая на дату «{period.label}»{part}; "
                        f"текст пункта: «{limit.text}»; вывод: {verdict}"
                    )
                    if line not in lines:
                        lines.append(line)
    return lines[:MAX_COMPARISONS]
