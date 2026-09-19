"""Freezing the suite: sha256 of the case and base files into `cases/FROZEN.md` (DESIGN 10.2).

Comments and docstrings translated to English for review; logic unchanged.

Usage: python experiments/freeze.py [--write]. Without --write it prints the
hashes and compares them with the ones already recorded.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness as h  # noqa: E402

FROZEN_PATH = h.SUITE_DIR / "cases" / "FROZEN.md"


def frozen_files() -> list[Path]:
    suite = h.load_suite()
    paths = [h.SUITE_DIR / "cases" / "cases.json"]
    for base in suite["bases"]:
        base_dir = h.SUITE_DIR / "bases" / base
        paths += [
            base_dir / "utterances.json",
            base_dir / "transcript.txt",
            base_dir / "reference.json",
        ]
    return paths


def hashes() -> list[tuple[str, str]]:
    return [
        (str(p.relative_to(h.SUITE_DIR)), h.sha256_file(p))
        for p in frozen_files()
    ]


def render_frozen(rows: list[tuple[str, str]], suite_name: str, frozen_on: str) -> str:
    lines = [
        f"# FROZEN: сьюта {suite_name}",
        "",
        f"Дата заморозки: {frozen_on}.",
        "",
        "Тексты баз, кейсов, филлеров, строка-канарейка и критерии успеха после этой",
        "даты не правятся. Правка означает новую версию сьюты и новый прогон целиком",
        "(DESIGN, раздел 10, пункт 2).",
        "",
        "| файл | sha256 |",
        "|---|---|",
    ]
    lines += [f"| `{name}` | `{digest}` |" for name, digest in rows]
    lines.append("")
    return "\n".join(lines)


def read_recorded() -> dict[str, str]:
    if not FROZEN_PATH.exists():
        return {}
    recorded: dict[str, str] = {}
    for line in FROZEN_PATH.read_text(encoding="utf-8").splitlines():
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) == 2 and parts[0].startswith("`") and parts[1].startswith("`"):
            recorded[parts[0].strip("`")] = parts[1].strip("`")
    return recorded


def main() -> int:
    suite = h.load_suite()
    rows = hashes()
    if "--write" in sys.argv:
        FROZEN_PATH.write_text(
            render_frozen(rows, suite["suite"], date.today().isoformat()), encoding="utf-8"
        )
        print(f"written: {FROZEN_PATH}")
        return 0

    recorded = read_recorded()
    if not recorded:
        print("FROZEN.md is missing or holds no hashes")
        for name, digest in rows:
            print(f"  {name}  {digest}")
        return 1
    bad = [name for name, digest in rows if recorded.get(name) != digest]
    for name, digest in rows:
        mark = "ok " if recorded.get(name) == digest else "MISMATCH"
        print(f"{mark} {name} {digest}")
    missing = sorted(set(recorded) - {name for name, _ in rows})
    for name in missing:
        print(f"MISMATCH {name}: recorded in FROZEN.md but not part of the suite")
    return 1 if bad or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
