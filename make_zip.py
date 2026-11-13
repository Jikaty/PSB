from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = [
    "config.py",
    "llm_client.py",
    "kb_loader.py",
    "chunker.py",
    "retriever.py",
    "prompts.py",
    "verifier.py",
    "pipeline.py",
    "solution.py",
    "requirements.txt",
]
ALIASES = {"requirements_start.txt": "requirements.txt"}


def main() -> int:
    out = ROOT / "submit.zip"
    missing = [name for name in FILES if not (ROOT / name).is_file()]
    if missing:
        print("Нет файлов: " + ", ".join(missing), file=sys.stderr)
        return 1
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(ROOT / name, arcname=name)
        for alias, name in ALIASES.items():
            archive.write(ROOT / name, arcname=alias)
    with zipfile.ZipFile(out) as archive:
        for info in archive.infolist():
            print(f"{info.filename}\t{info.file_size}")
    print(f"Итого: {out.name}, {out.stat().st_size} байт")
    return 0


if __name__ == "__main__":
    sys.exit(main())
