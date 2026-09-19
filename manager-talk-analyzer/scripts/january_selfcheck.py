#!/usr/bin/env python
"""Self-check of the January extraction batch: a separate model call, "transcript vs JSON".

Comments and docstrings translated to English for review; logic unchanged.
The three contradiction type names in ``TYPES`` are data values that also appear
verbatim in the prompt, so they were translated together with it; every
comparison that uses them was updated in step. Nothing else changed.

The prompt is NOT the extraction prompt: the model does not re-extract facts, it
looks for contradictions between a finished extraction and the transcript.
Temperature 0, answer constrained by a schema.

Read-only: ``documents.result`` is never modified. The synthetic corruption used
to control the method is applied to a copy of the dict in memory and never
reaches the database.

    .venv/bin/python scripts/january_selfcheck.py [--limit N] [--seed 20260811]
                                                  [--out PATH] [--control-only]
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mta import db  # noqa: E402
from mta.config import load_config  # noqa: E402
from mta.ollama import OllamaClient, OllamaError  # noqa: E402

SEED = 20260811
STRATA = [("in", 40), ("out", 30)]
TOP_SCORE = 10

TYPES = ("fabrication", "omission", "distortion")

SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["contradictions"],
    "properties": {
        "contradictions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "field", "detail"],
                "properties": {
                    "type": {"type": "string", "enum": list(TYPES)},
                    "field": {"type": "string"},
                    "detail": {"type": "string"},
                },
            },
        }
    },
}

SYSTEM = """You are a meticulous reviewer. You are given the transcript of a phone
conversation and a finished JSON extraction of that same conversation. Your task is to
find contradictions between them.

Types of contradiction:
- "fabrication": the JSON states a fact (an amount, a name, a company, an agreement,
  an objection) that is not present in the transcript;
- "omission": the transcript contains something essential (a client request, an
  agreement, a named amount, an objection) and the JSON does not have it;
- "distortion": the fact is present in both, but the JSON distorts it - a different
  amount, a different name, the sides swapped, the meaning of an agreement changed.

Rules:
- report only what is visible in the text; do not speculate and do not build hypotheses;
- do not nitpick wording, abbreviations or the brevity of the summary;
- the field "field" is the name of the JSON field the contradiction belongs to;
- the field "detail" is one sentence saying what exactly is wrong;
- if there are no contradictions, return an empty contradictions array."""


# Service fields are filled by code from facts about the source (file name, folder,
# formula), they are not extracted from the conversation. We do not show them to the
# reviewer: it cannot confirm them from the transcript and honestly flags them as
# "fabrication", which is noise rather than findings.
SERVICE_FIELDS = (
    "source_kind",
    "employee_id",
    "identity",
    "source_ref",
    "contact_type",
    "score",
    "score_breakdown",
    "missing_fields",
)


def content_view(result: dict[str, Any]) -> dict[str, Any]:
    """Only what the model actually extracted from the conversation."""
    return {k: v for k, v in result.items() if k not in SERVICE_FIELDS}


def user_prompt(transcript: str, result: dict[str, Any]) -> str:
    # The schema is duplicated as text: the nginx proxy in front of Ollama does not
    # apply the format parameter, so `format` alone does not guarantee the shape of
    # the answer (the same reason as in prompts._schema_block).
    return (
        "TRANSCRIPT:\n"
        f"{transcript}\n\n"
        "JSON EXTRACTION:\n"
        f"{json.dumps(content_view(result), ensure_ascii=False, indent=1)}\n\n"
        "Find the contradictions between the extraction and the transcript.\n\n"
        "The answer is ONE JSON object and nothing else, strictly following the schema "
        "below (an object with a contradictions key, not a bare array).\n"
        "JSON SCHEMA OF THE ANSWER:\n" + json.dumps(SCHEMA, ensure_ascii=False, indent=1)
    )


def contradictions_of(answer: Any) -> list[dict[str, Any]]:
    """Pulls the list of contradictions out of the answer, tolerating a bare array."""
    if isinstance(answer, dict):
        found = answer.get("contradictions")
    elif isinstance(answer, list):
        found = answer
    else:
        found = None
    if not isinstance(found, list):
        return []
    return [c for c in found if isinstance(c, dict)]


def corrupt(result: dict[str, Any], kind: str) -> tuple[dict[str, Any], str, str, str]:
    """Synthetic corruption of a copy of the extraction, to control the method.

    Returns (corrupted copy, description, target field, marker to look for in the answer).
    """
    spoiled = copy.deepcopy(result)
    if kind == "amount":
        amounts = spoiled.get("amounts_mentioned") or []
        if amounts:
            old = amounts[0].get("value")
            amounts[0]["value"] = round(float(old) * 7 + 13, 2)
            return (
                spoiled,
                f"amount {old} replaced with {amounts[0]['value']}",
                "amounts_mentioned",
                str(amounts[0]["value"]).rstrip("0").rstrip("."),
            )
        spoiled["amounts_mentioned"] = [
            {"value": 987654.0, "currency": "RUB", "context": "prepayment for the batch"}
        ]
        return (
            spoiled,
            "a non-existent amount of 987654 RUB was added",
            "amounts_mentioned",
            "987654",
        )
    if kind == "objection":
        spoiled.setdefault("objections", []).append(
            {
                "objection": "The client said a competitor is 40 % cheaper",
                "handled": False,
                "how": None,
            }
        )
        return spoiled, "a false objection about a competitor was added", "objections", "competitor"
    if kind == "name":
        participants = spoiled.setdefault("participants", {})
        old = participants.get("client")
        participants["client"] = "Benjamin Sidorchuk"
        return (
            spoiled,
            f"client name {old!r} replaced with an invented one",
            "participants.client",
            "Benjamin",
        )
    raise ValueError(kind)


def caught(found: list[dict[str, Any]], field: str, marker: str) -> bool:
    """Was the injected corruption caught: by the target field or by the marker in the text."""
    target = field.split(".")[-1].casefold()
    needle = marker.casefold()
    for item in found:
        text = f"{item.get('field', '')} {item.get('detail', '')}".casefold()
        if needle and needle in text:
            return True
        if target in text and item.get("type") in ("fabrication", "distortion"):
            return True
    return False


CONTROL_KINDS = ["amount", "objection", "name", "amount", "objection"]


def pick(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Stratified sample with a fixed seed."""
    rnd = random.Random(seed)
    chosen: list[dict[str, Any]] = []
    used: set[int] = set()
    for direction, count in STRATA:
        pool = [r for r in rows if r["call_direction"] == direction]
        rnd.shuffle(pool)
        take = pool[:count]
        chosen.extend(take)
        used.update(r["id"] for r in take)
    by_score = sorted(
        (r for r in rows if r["id"] not in used and r["result"].get("score") is not None),
        key=lambda r: (-r["result"]["score"], r["id"]),
    )
    chosen.extend(by_score[:TOP_SCORE])
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="cap the number of documents (debugging)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default="selfcheck.json")
    parser.add_argument("--control-only", action="store_true", help="only the corruption control")
    args = parser.parse_args()

    cfg = load_config()
    client = OllamaClient(cfg)
    with db.connect(cfg.database_url) as conn:
        rows = [
            dict(r)
            for r in conn.execute(
                """
                SELECT id, source_file, employee, call_direction, duration_s, transcript, result
                FROM documents
                WHERE kind = 'audio' AND status = 'done' AND result IS NOT NULL
                  AND transcript IS NOT NULL
                  AND communication_at >= '2026-01-01' AND communication_at < '2026-02-01'
                ORDER BY id
                """
            ).fetchall()
        ]

    sample = pick(rows, args.seed)
    if args.limit:
        sample = sample[: args.limit]
    out_path = Path(args.out)
    report: dict[str, Any] = {
        "seed": args.seed,
        "model": cfg.ollama_model,
        "sample_ids": [r["id"] for r in sample],
        "items": [],
        "control": [],
    }

    def ask(transcript: str, result: dict[str, Any]) -> tuple[Any, float, str | None]:
        started = time.time()
        try:
            answer = client.chat_json(
                system=SYSTEM,
                user=user_prompt(transcript, result),
                schema=SCHEMA,
                temperature=0.0,
            )
            return answer, time.time() - started, None
        except OllamaError as exc:
            return None, time.time() - started, str(exc)

    if not args.control_only:
        print(f"sample: {len(sample)} documents, seed {args.seed}, model {cfg.ollama_model}")
        for n, row in enumerate(sample, 1):
            answer, seconds, error = ask(row["transcript"], row["result"])
            found = contradictions_of(answer)
            report["items"].append(
                {
                    "id": row["id"],
                    "file": Path(row["source_file"]).name,
                    "employee": row["employee"],
                    "direction": row["call_direction"],
                    "duration_s": row["duration_s"],
                    "seconds": round(seconds, 1),
                    "error": error,
                    "contradictions": found,
                }
            )
            print(
                f"[{n}/{len(sample)}] id={row['id']:5d} {row['call_direction']:3s} "
                f"{row['duration_s'] or 0:6.0f}s contradictions={len(found)} {seconds:5.1f}s"
                + (f" ERROR: {error[:60]}" if error else ""),
                flush=True,
            )
            out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))

    # Control of the method: five documents with synthetic corruption of a copy of
    # the extraction.
    control_rows = [r for r in sample if r["result"].get("summary")][:5]
    print(f"\ncorruption control: {len(control_rows)} documents")
    hits = 0
    for row, kind in zip(control_rows, CONTROL_KINDS):
        spoiled, what, field, marker = corrupt(row["result"], kind)
        answer, seconds, error = ask(row["transcript"], spoiled)
        found = contradictions_of(answer)
        is_caught = caught(found, field, marker)
        hits += int(is_caught)
        report["control"].append(
            {
                "id": row["id"],
                "kind": kind,
                "injected": what,
                "field": field,
                "caught": is_caught,
                "seconds": round(seconds, 1),
                "error": error,
                "contradictions": found,
            }
        )
        print(
            f"  id={row['id']:5d} corruption[{kind}]: {what[:55]} -> "
            f"{'CAUGHT' if is_caught else 'NOT CAUGHT'} (contradictions in total {len(found)}, "
            f"{seconds:.1f} s)" + (f" ERROR: {error[:60]}" if error else ""),
            flush=True,
        )
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))

    if control_rows:
        report["control_caught"] = [hits, len(control_rows)]
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))
        print(f"control: caught {hits} of {len(control_rows)}")
    print(f"\nresult: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
