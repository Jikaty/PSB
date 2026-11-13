from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

from config import FLAG_DEFAULTS, FLAG_OFF, FLAG_ON, flags_text

ROOT = Path(__file__).resolve().parent
FILES = [
    "config.py",
    "llm_client.py",
    "kb_loader.py",
    "chunker.py",
    "retriever.py",
    "prompts.py",
    "verifier.py",
    "timeline.py",
    "pipeline.py",
    "solution.py",
    "requirements.txt",
]
ALIASES = {"requirements_start.txt": "requirements.txt"}
FLAGS_FILE = "flags.py"


def parse_flags(items: list[str]) -> dict[str, bool]:
    values = dict(FLAG_DEFAULTS)
    for item in (part for chunk in items for part in chunk.split(",") if part.strip()):
        name, sep, raw = item.strip().partition("=")
        name, raw = name.strip().upper(), raw.strip().lower()
        if name not in FLAG_DEFAULTS:
            raise ValueError(f"неизвестный флаг {name}; есть: {', '.join(FLAG_DEFAULTS)}")
        if not sep or raw not in FLAG_ON + FLAG_OFF:
            raise ValueError(f"флаг {name}: нужно значение 1 или 0")
        values[name] = raw in FLAG_ON
    return values


def flags_source(values: dict[str, bool]) -> str:
    return "".join(f"{name} = {value}\n" for name, value in values.items())


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout


def commit() -> str:
    try:
        head = git("rev-parse", "--short", "HEAD").strip()
        changed = [line[3:].strip().strip('"') for line in git("status", "--porcelain").splitlines()]
    except (OSError, subprocess.SubprocessError):
        return "неизвестен"
    dirty = any(name in FILES for name in changed)
    return (head or "неизвестен") + (" + незакоммиченные правки" if dirty else "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flags", nargs="*", metavar="USE_X=0")
    parser.add_argument("--out", default="submit.zip")
    args = parser.parse_args()
    try:
        values = None if args.flags is None else parse_flags(args.flags)
    except ValueError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    if (ROOT / FLAGS_FILE).exists():
        print(f"Ошибка: в корне лежит {FLAGS_FILE}; удалите его, флаги задаются через --flags", file=sys.stderr)
        return 1
    missing = [name for name in FILES if not (ROOT / name).is_file()]
    if missing:
        print("Нет файлов: " + ", ".join(missing), file=sys.stderr)
        return 1
    out = ROOT / Path(args.out).name
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(ROOT / name, arcname=name)
        for alias, name in ALIASES.items():
            archive.write(ROOT / name, arcname=alias)
        if values is not None:
            archive.writestr(FLAGS_FILE, flags_source(values))
    with zipfile.ZipFile(out) as archive:
        for info in archive.infolist():
            print(f"{info.filename}\t{info.file_size}")
    print(f"Итого: {out.name}, {out.stat().st_size} байт")
    print(f"Коммит: {commit()}")
    note = "" if values is not None else " (по умолчанию, без flags.py)"
    print(f"Флаги: {flags_text(values or FLAG_DEFAULTS)}{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
