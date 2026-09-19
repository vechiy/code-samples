#!/usr/bin/env python3
"""Scoring an ASR + diarization run against the golden reference.

Comments and docstrings translated to English for review; logic unchanged.

Input: a run JSON with words and diarization segments (shape
``{"words": [{start,end,text}], "segments": [{start,end,speaker}]}``) or a
document from the database via ``--doc-id``. Utterances are assembled by the
regular pipeline path ``pipeline.build_reply_utterances``, that is, we score
exactly what the LLM will see.

Metrics (see tests/golden/golden_*.json):
- attribution_accuracy_time - the share of reference segment time given to the
  correct speaker (by majority of overlap), under the better of the two label
  mappings;
- coverage_lost_zones - in how many of the six loss zones words appeared;
- WER by text_status group (owner_confirmed, from_service, corrected_by_pitch);
- anomalies - utterances faster than 6 words/s and utterances more than half of
  whose interval falls on silence.

Segments with status unknown/ambiguous do not take part in WER (there is no text
for them in the reference). A token in angle brackets (``<ИМЯ>``) is an
anonymisation mask: in WER it matches any word of the run, otherwise masking
would count as a recognition error.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from itertools import permutations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

SPEED_LIMIT_WPS = 6.0          # speed anomaly threshold, words per second
SILENCE_SHARE_LIMIT = 0.5      # share of the utterance interval spent on silence
WER_STATUSES = ("owner_confirmed", "from_service", "corrected_by_pitch")
WILDCARD = "\x00*"                # internal "any word" marker for WER
WILDCARD_RE = re.compile(r"<[^<>\s]+>")   # anonymisation mask in the reference text
FRAME_S = 0.02
SILENCE_MARGIN_DB = 8.0        # speech threshold = noise floor plus this many dB


# --------------------------------------------------------------------------- input

def load_run(path: Path) -> tuple[list[dict], list[dict] | None]:
    data = json.loads(path.read_text(encoding="utf-8"))
    words = [
        {"start": float(w["start"]), "end": float(w["end"]), "text": str(w["text"]).strip()}
        for w in data.get("words") or []
        if str(w.get("text", "")).strip()
    ]
    segments = data.get("segments")
    if segments is not None:
        segments = sorted(
            (
                {"start": float(s["start"]), "end": float(s["end"]), "speaker": str(s["speaker"])}
                for s in segments
            ),
            key=lambda s: s["start"],
        )
    return words, segments


def load_doc(doc_id: int) -> tuple[list[dict], list[dict] | None]:
    import os

    import psycopg
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        cur.execute("SELECT words, speaker_segments FROM documents WHERE id = %s", (doc_id,))
        row = cur.fetchone()
    if row is None:
        raise SystemExit(f"document {doc_id} is not in the database")
    words, segments = row
    if not words:
        raise SystemExit(
            f"document {doc_id} has an empty words column: it was processed before that"
            " column existed (ASR artifacts were not stored). Score a run of the file"
            " via --run instead."
        )
    return words, segments


def build_utterances(words: list[dict], segments: list[dict] | None) -> list[dict]:
    """Utterances assembled by the regular pipeline path."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    from mta import asr, diarize, pipeline
    from mta.config import load_config

    w = [asr.Word(start=x["start"], end=x["end"], text=x["text"]) for x in words]
    segs = [
        diarize.SpeakerSegment(start=s["start"], end=s["end"], speaker=s["speaker"])
        for s in (segments or [])
    ]
    utts = pipeline.build_reply_utterances(load_config(), w, segs)
    return [{"speaker": u.speaker, "start": u.start, "end": u.end, "text": u.text} for u in utts]


# ------------------------------------------------------------------- signal/silence

def speech_mask(wav: Path) -> tuple[list[bool], float]:
    """Frame-level speech mask by energy. Needed for the "utterance over silence" anomaly."""
    import numpy as np

    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(wav), "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
        capture_output=True,
    ).stdout
    if not raw:
        raise SystemExit(f"ffmpeg could not read {wav}")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    n = int(8000 * FRAME_S)
    m = len(x) // n
    fr = x[: m * n].reshape(m, n)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1) + 1e-12) + 1e-12)
    thr = float(np.percentile(db, 10)) + SILENCE_MARGIN_DB
    return (db > thr).tolist(), thr


def silence_share(mask: list[bool], start: float, end: float) -> float:
    i0, i1 = int(start / FRAME_S), max(int(start / FRAME_S) + 1, int(end / FRAME_S))
    window = mask[i0:i1]
    if not window:
        return 1.0
    return 1.0 - sum(window) / len(window)


# ---------------------------------------------------------------------- metrics

def _norm_plain(text: str) -> list[str]:
    keep = [ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text]
    return "".join(keep).replace("ё", "е").split()


def norm_words(text: str | None) -> list[str]:
    """Words of the text. An anonymisation mask ``<ИМЯ>`` yields a single wildcard
    token: plain normalisation would eat the brackets and turn the mask into a word."""
    if not text:
        return []
    out: list[str] = []
    pos = 0
    for m in WILDCARD_RE.finditer(text):
        out.extend(_norm_plain(text[pos:m.start()]))
        out.append(WILDCARD)
        pos = m.end()
    out.extend(_norm_plain(text[pos:]))
    return out


def wer(ref: list[str], hyp: list[str]) -> tuple[int, int]:
    """(number of errors, reference length) by word-level Levenshtein distance.

    A wildcard in the reference (the anonymisation mask) matches any word of the
    run, so a substitution on it costs 0. Deleting a wildcard is still an error:
    it means nothing at all showed up at that place in the run.
    """
    if not ref:
        return (len(hyp), 0)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            sub = 0 if (r == h or r == WILDCARD) else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + sub))
        prev = cur
    return prev[-1], len(ref)


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def midpoint_in(word: dict, start: float, end: float) -> bool:
    """A word belongs to a zone by its midpoint: whisper inflates word ends with pauses."""
    mid = (word["start"] + word["end"]) / 2
    return start <= mid < end


def attribution(gold: dict, utts: list[dict]) -> tuple[float, dict[str, str], float]:
    """Share of reference time given to the correct speaker, under the best label mapping."""
    labels = sorted({u["speaker"] for u in utts})
    gold_ids = sorted({g["speaker"] for g in gold["segments"]})
    best = (0.0, {})
    total = sum(g["end"] - g["start"] for g in gold["segments"])
    for perm in permutations(labels, min(len(labels), len(gold_ids))):
        mapping = dict(zip(perm, gold_ids))
        hit = 0.0
        for g in gold["segments"]:
            for u in utts:
                if mapping.get(u["speaker"]) == g["speaker"]:
                    hit += overlap(u["start"], u["end"], g["start"], g["end"])
        if hit > best[0]:
            best = (hit, mapping)
    return (best[0] / total if total else 0.0), best[1], total


def collapse_speakers(seq: list[str]) -> list[str]:
    """Collapses adjacent identical labels: we compare the flow of the dialogue, not the slicing."""
    out: list[str] = []
    for item in seq:
        if not out or out[-1] != item:
            out.append(item)
    return out


def structure(gold: dict, utts: list[dict], mapping: dict[str, str]) -> dict[str, Any]:
    """Criterion "the sequence of utterance speakers matches the reference".

    Both sequences are collapsed first: if the pipeline split one turn into two
    utterances of the same speaker, that is not a divergence of the dialogue flow.

    A relaxed variant of the reference is computed separately, without turns marked
    ``resolution = unresolvable_mono``. Those are turns the owner ruled unresolvable
    on a mono recording (the measurement of the limit lives in the reference itself,
    in the ``resolution_note`` field). Dropping such a turn collapses its neighbours
    if they belong to the same speaker, so the variant is computed by the same rules.
    """
    ref = collapse_speakers([g["speaker"] for g in gold["segments"]])
    hyp = collapse_speakers([mapping.get(u["speaker"], u["speaker"]) for u in utts])
    first = next((i for i, (a, b) in enumerate(zip(ref, hyp)) if a != b), None)
    if first is None and len(ref) != len(hyp):
        first = min(len(ref), len(hyp))
    excused = [g["id"] for g in gold["segments"]
               if g.get("resolution") == "unresolvable_mono"]
    ref_relaxed = collapse_speakers(
        [g["speaker"] for g in gold["segments"]
         if g.get("resolution") != "unresolvable_mono"]
    )
    return {
        "ok": ref == hyp,
        "ok_relaxed": ref_relaxed == hyp,
        "ref_turns": len(ref),
        "hyp_turns": len(hyp),
        "relaxed_turns": len(ref_relaxed),
        "excused_ids": excused,
        "ref": "".join(x[-1] for x in ref),
        "ref_relaxed": "".join(x[-1] for x in ref_relaxed),
        "hyp": "".join(x[-1] for x in hyp),
        "first_diff_turn": first,
    }


def score(gold: dict, words: list[dict], segments: list[dict] | None,
          utts: list[dict], mask: list[bool] | None) -> dict[str, Any]:
    lost_ids = set(gold["lost_zones_ids"])
    by_zone = []
    covered = 0
    for g in gold["segments"]:
        zw = [w for w in words if midpoint_in(w, g["start"], g["end"])]
        if g["id"] in lost_ids and zw:
            covered += 1
        by_zone.append({"id": g["id"], "speaker": g["speaker"], "start": g["start"],
                        "end": g["end"], "status": g["text_status"],
                        "n_words": len(zw), "hyp": " ".join(w["text"] for w in zw)})

    per_status: dict[str, list[int]] = {s: [0, 0] for s in WER_STATUSES}
    for g, z in zip(gold["segments"], by_zone):
        if g["text_status"] not in per_status:
            continue
        err, n = wer(norm_words(g["text"]), norm_words(z["hyp"]))
        per_status[g["text_status"]][0] += err
        per_status[g["text_status"]][1] += n
    wer_by_status = {
        s: (round(e / n, 3) if n else None) for s, (e, n) in per_status.items()
    }
    tot_e = sum(e for e, _ in per_status.values())
    tot_n = sum(n for _, n in per_status.values())

    acc, mapping, gold_time = attribution(gold, utts)
    struct = structure(gold, utts, mapping)

    speed, silent = [], []
    for u in utts:
        dur = u["end"] - u["start"]
        n = len(u["text"].split())
        if dur > 0 and n / dur > SPEED_LIMIT_WPS:
            speed.append({"start": round(u["start"], 2), "end": round(u["end"], 2),
                          "wps": round(n / dur, 1), "text": u["text"][:60]})
        if mask is not None:
            sh = silence_share(mask, u["start"], u["end"])
            if sh > SILENCE_SHARE_LIMIT:
                silent.append({"start": round(u["start"], 2), "end": round(u["end"], 2),
                               "silence": round(sh, 2), "text": u["text"][:60]})

    return {
        "n_words": len(words),
        "n_segments": len(segments) if segments is not None else None,
        "n_utterances": len(utts),
        "coverage_lost_zones": f"{covered}/{len(lost_ids)}",
        "coverage_lost_zones_n": covered,
        "attribution_accuracy_time": round(acc, 3),
        "structure": struct,
        "speaker_map": mapping,
        "gold_speech_s": round(gold_time, 1),
        "wer_total": round(tot_e / tot_n, 3) if tot_n else None,
        "wer_by_status": wer_by_status,
        "wer_ref_words": tot_n,
        "anomalies_speed": speed,
        "anomalies_silence": silent,
        "zones": by_zone,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="run JSON with words and segments")
    src.add_argument("--doc-id", type=int, help="document from the database")
    ap.add_argument("--gold", default=str(ROOT / "tests/golden" / "golden_call-01.json"))
    ap.add_argument("--segments-from", help="take diarization segments from another run"
                                            " (when this run computed ASR only)")
    ap.add_argument("--wav", default=None,
                    help="source audio for the \"utterance over silence\" detector; the path"
                         " to the golden case recording is in analytic/golden_map.md (outside"
                         " the repository), without --wav the detector is skipped")
    ap.add_argument("--label", default=None)
    ap.add_argument("--json", action="store_true", help="JSON only, no tables")
    ap.add_argument("--zones", action="store_true", help="print the per-zone breakdown")
    args = ap.parse_args()

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    if args.run:
        words, segments = load_run(Path(args.run))
        label = args.label or Path(args.run).stem
    else:
        words, segments = load_doc(args.doc_id)
        label = args.label or f"doc{args.doc_id}"
    if segments is None and args.segments_from:
        _, segments = load_run(Path(args.segments_from))

    utts = build_utterances(words, segments)
    mask = None
    if args.wav:
        mask, _ = speech_mask(Path(args.wav))

    result = score(gold, words, segments, utts, mask)
    result["label"] = label

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=1))
        return

    print(f"=== {label} ===")
    print(f"words {result['n_words']}, diarization segments {result['n_segments']}, "
          f"utterances {result['n_utterances']}")
    st = result["structure"]
    print(f"utterance structure      {'MATCHES' if st['ok'] else 'DIVERGES'}"
          f"  (turns: reference {st['ref_turns']}, run {st['hyp_turns']}"
          + (f", first divergence at turn {st['first_diff_turn'] + 1}"
             if not st["ok"] and st["first_diff_turn"] is not None else "") + ")")
    print(f"  reference {st['ref']}")
    print(f"  run       {st['hyp']}")
    print(f"coverage_lost_zones      {result['coverage_lost_zones']}")
    print(f"attribution_accuracy_time {result['attribution_accuracy_time']}  "
          f"(labels: {result['speaker_map']})")
    print(f"WER total                {result['wer_total']}  "
          f"(reference words {result['wer_ref_words']})")
    for s, v in result["wer_by_status"].items():
        print(f"  WER {s:<20} {v}")
    print(f"speed anomalies > {SPEED_LIMIT_WPS:.0f} words/s: {len(result['anomalies_speed'])}")
    for a in result["anomalies_speed"]:
        print(f"    {a['start']:.2f}-{a['end']:.2f} {a['wps']} w/s | {a['text']}")
    print(f"\"utterance over silence\" anomalies: {len(result['anomalies_silence'])}")
    for a in result["anomalies_silence"]:
        print(f"    {a['start']:.2f}-{a['end']:.2f} silence {a['silence']:.0%} | {a['text']}")
    if args.zones:
        print(f"\n{'id':>3} {'zone':<14}{'sp':<4}{'status':<19}{'words':>5}  run text")
        for z in result["zones"]:
            print(f"{z['id']:>3} {z['start']:>6.2f}-{z['end']:<6.2f}{z['speaker']:<4}"
                  f"{z['status']:<19}{z['n_words']:>5}  {z['hyp'][:60]}")


if __name__ == "__main__":
    main()
