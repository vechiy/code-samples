#!/usr/bin/env python
"""Excerpt: blind A/B material for two ASR engines, plus the metric it is read with.

Comments and docstrings translated to English for review; logic unchanged.

WHERE IT COMES FROM
    scripts/asr_ab_compare.py, lines 225-315 -> build_blind()
    scripts/asr_ab_lib.py,     lines 168-207 -> normalize(), levenshtein(), error_rate()

WHAT WAS OMITTED
    - the database layer (the asr_ab table, upsert, reading document pairs);
    - the GigaAM wrapper with its longform VAD slicing;
    - config loading (AbConfig, .env and scripts/asr_ab.env), so the `cfg` object
      arrives here as a parameter: it supplies blind_seed, blind_top and blind_random;
    - per-file CSV writing and the summary.md generator;
    - the CLI entry point.

WHAT CHANGED RELATIVE TO THE ORIGINAL
    - `L.normalize(...)` became `normalize(...)`, because the shared module is not
      imported in this excerpt (two call sites inside build_blind);
    - nothing else. The `cfg: L.AbConfig` annotation is kept verbatim and stays
      valid because of `from __future__ import annotations`: annotations are never
      evaluated here.

WHY THESE THREE FUNCTIONS TRAVEL TOGETHER
    build_blind is what makes the comparison blind; error_rate is the number the
    reviewer sees next to each case in the key. Reading one without the other is
    misleading: this WER is the DIVERGENCE of two engines, not the error of either
    one. There is no reference text for these recordings at all, and the engine that
    happens to be in production is in the denominator only because it is the
    incumbent.
"""

from __future__ import annotations

import random
import re
import shutil
from pathlib import Path


# ------------------------------------------------------------------- texts

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lower case, ё to е, no punctuation, single spaces.

    GigaAM returns text without punctuation and without capitals, so this is the
    only shape in which the two engines can be compared at all.
    """
    text = text.lower().replace("ё", "е")
    text = text.replace("-", " ").replace("—", " ").replace("–", " ")
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


# ------------------------------------------------------------ WER/CER metrics

def levenshtein(a: list[str] | str, b: list[str] | str) -> int:
    """Edit distance. Lists give WER, strings give CER."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def error_rate(ref: list[str] | str, hyp: list[str] | str) -> float:
    """Share of edits relative to ref. Empty ref: 0 when hyp is empty, otherwise 1."""
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


# -------------------------------------------------------- blind review material

def build_blind(rows: list[dict], cfg: L.AbConfig, out_dir: Path,
                pairs_by_hash: dict[str, dict]) -> None:
    """6 cases: the top divergences plus random ones. The A/B order is randomised."""
    long_enough = [r for r in rows if r["words_whisper"] >= 30 and r["words_gigaam"] >= 30]
    ranked = sorted(long_enough, key=lambda r: r["wer"], reverse=True)
    top = ranked[: cfg.blind_top]

    rnd = random.Random(cfg.blind_seed)
    top_hashes = {r["file_hash"] for r in top}
    rest = [r for r in long_enough if r["file_hash"] not in top_hashes]
    picked_random = rnd.sample(rest, min(cfg.blind_random, len(rest)))
    cases = top + picked_random

    # Who goes as variant A: the layout is split exactly in half and only then
    # shuffled. A plain coin flip over six cases easily produces a 5:1 imbalance,
    # and the engine can then be guessed without listening to anything.
    first = ["whisper", "gigaam"] * ((len(cases) + 1) // 2)
    first = first[: len(cases)]
    rnd.shuffle(first)

    blind_dir = out_dir / "cases"
    if blind_dir.exists():
        shutil.rmtree(blind_dir)
    blind_dir.mkdir(parents=True)

    key_lines = [
        "# KEY to the blind A/B review of ASR engines",
        "",
        "Do not open until the assessment has been filled in in cases/README.md.",
        "",
        "| case | variant A | variant B | file | WER | selected as |",
        "|---|---|---|---|---|---|",
    ]
    readme = [
        "# Blind ASR review",
        "",
        "For each case: listen to the audio and mark which of the two variants is",
        "closer to what you hear. Which engine is where is in a separate file, KEY.md;",
        "open it after the assessment.",
        "",
        "Both texts are reduced to the same shape: lower case, no punctuation.",
        "One of the engines has no punctuation at all, so otherwise the choice would",
        "have been obvious for reasons other than recognition quality.",
        "",
        "One difference could not be hidden: one engine writes numbers as digits",
        "(\"1200\"), the other as words (\"one thousand two hundred\"). The engine can be",
        "guessed from that, but by itself it says nothing about whether the number was",
        "heard correctly. Judge the words, not the way digits are written.",
        "",
    ]

    for n, r in enumerate(cases, start=1):
        pair = pairs_by_hash[r["file_hash"]]
        src = Path(pair["source_file"])
        case_dir = blind_dir / f"case{n}"
        case_dir.mkdir()

        audio_dst = case_dir / f"audio{src.suffix}"
        if src.exists():
            shutil.copy2(src, audio_dst)
            audio_note = audio_dst.name
        else:
            audio_note = "FILE NOT FOUND ON DISK"

        texts = {"whisper": normalize(r["whisper_text"]),
                 "gigaam": normalize(r["gigaam_text"])}
        a_engine = first[n - 1]
        b_engine = "gigaam" if a_engine == "whisper" else "whisper"
        variants = [(a_engine, texts[a_engine]), (b_engine, texts[b_engine])]

        (case_dir / "A.txt").write_text(variants[0][1] + "\n", encoding="utf-8")
        (case_dir / "B.txt").write_text(variants[1][1] + "\n", encoding="utf-8")

        origin = "top divergence" if r in top else "random"
        readme += [
            f"## Case {n}",
            "",
            f"- audio: `cases/case{n}/{audio_note}` ({r['duration_s']:.0f} s)",
            f"- variant A: `cases/case{n}/A.txt`",
            f"- variant B: `cases/case{n}/B.txt`",
            "- closer to what you hear: [ ] A  [ ] B  [ ] the same",
            "- notes:",
            "",
        ]
        key_lines.append(
            f"| {n} | {variants[0][0]} | {variants[1][0]} | {r['file']} | "
            f"{r['wer']:.3f} | {origin} |"
        )

    (blind_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    (out_dir / "KEY.md").write_text("\n".join(key_lines) + "\n", encoding="utf-8")
