"""selfcheck v2: the contradiction judge with answer validation, retries and a corruption control.

Comments and docstrings translated to English for review; logic unchanged.

v1 (`scripts/january_selfcheck.py`) is neither modified nor deleted: the prompt,
the answer schema, the view of the parse shown to the model, the v1 corruptions
and the "was the corruption caught" predicate are imported from it. Only the
handling of the model answer changes:

1. the answer is validated against the schema (`jsonschema`); a bare array and
   extra fields are no longer accepted, a malformed answer is an error rather
   than an empty list of findings;
2. on a parse or validation failure up to two retries are made with the error
   text in the prompt, as in `pipeline._extract`; the count of unparsed answers
   goes into the report;
3. corruption control: 10 calls against three kinds of corruption (replacing a
   value, deleting a fact, distorting a number, a date or a name);
4. the denominator for false positives: the same judge on the untouched parses of
   the same 10 calls.

The judge prompt, the answer schema, the model and the temperature are unchanged.

    python experiments/selfcheck_v2.py [--limit N] [--out PATH] [--compare-v1]
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

os.environ["MTA_HOME"] = os.environ.get("MTA_ROOT", "/opt/manager-talk-analyzer")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness as h  # noqa: E402

from jsonschema import Draft7Validator  # noqa: E402
from mta import db  # noqa: E402
from mta.config import load_config  # noqa: E402
from mta.ollama import OllamaError, _extract_json  # noqa: E402

MTA_ROOT = Path(os.environ["MTA_HOME"])
V1_PATH = MTA_ROOT / "scripts" / "january_selfcheck.py"
V1_REPORT = MTA_ROOT / "reports" / "2026-08-11_selfcheck-january.json"
OUT_DEFAULT = h.SUITE_DIR.parent / "analytic" / "mats-selfcheck" / "selfcheck_v2.json"
SEED = 20260919
RETRIES = 2
JUDGE_TEMPERATURE = 0.0
KINDS = ("replace", "delete", "distort")

SAMPLE_SQL = """
SELECT id, source_file, employee, call_direction, duration_s, transcript, result
FROM documents
WHERE status = 'done' AND kind = 'audio' AND duration_s BETWEEN 60 AND 300
  AND processed_at >= now() - interval '3 months'
  AND coalesce(result->>'contact_type', '') <> 'внутренний разговор'
  AND transcript IS NOT NULL
ORDER BY id
"""


def load_v1():
    spec = importlib.util.spec_from_file_location("january_selfcheck", V1_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------- judge

def validate(answer: Any) -> str | None:
    """A malformed answer is an error. v1 tolerated a bare array here."""
    v1 = load_v1()
    if not isinstance(answer, dict):
        return f"answer is not an object but {type(answer).__name__}"
    errors = sorted(Draft7Validator(v1.SCHEMA).iter_errors(answer), key=lambda e: list(e.path))
    if not errors:
        return None
    first = errors[0]
    where = "/".join(str(p) for p in first.absolute_path) or "<root>"
    return f"{where}: {first.message}"


def ask(cfg, v1, transcript: str, result: dict[str, Any]) -> dict[str, Any]:
    """Judge call with schema retries, as in pipeline._extract."""
    retry_error: str | None = None
    attempts: list[dict[str, Any]] = []
    for attempt in range(RETRIES + 1):
        user = v1.user_prompt(transcript, result)
        if retry_error:
            # The retry block stays in Russian: it is appended to the judge prompt,
            # and the prompt is not translated. Gloss: "WARNING: your previous answer
            # failed JSON-schema validation. Validator error: ... Fix the answer and
            # return JSON that fully conforms to the schema."
            user += ("\n\nВНИМАНИЕ: предыдущий твой ответ не прошёл валидацию по JSON-схеме."
                     f"\nОшибка валидатора: {retry_error}"
                     "\nИсправь ответ и верни JSON, полностью соответствующий схеме.")
        started = time.time()
        record: dict[str, Any] = {"attempt_no": attempt + 1, "seconds": None,
                                  "error": None, "validation_error": None}
        try:
            data = h.chat_raw(cfg, system=v1.SYSTEM, user=user, fmt=v1.SCHEMA,
                              options={"temperature": JUDGE_TEMPERATURE})
            record["seconds"] = round(time.time() - started, 1)
            content = (data.get("message") or {}).get("content", "")
            record["tokens"] = {"prompt_eval_count": data.get("prompt_eval_count"),
                                "eval_count": data.get("eval_count")}
            if not content.strip():
                raise OllamaError("Ollama returned empty content")
            answer = _extract_json(content)
        except OllamaError as exc:
            record["seconds"] = record["seconds"] or round(time.time() - started, 1)
            record["error"] = str(exc)
            retry_error = str(exc)
            attempts.append(record)
            continue
        problem = validate(answer)
        record["validation_error"] = problem
        attempts.append(record)
        if problem is None:
            return {"ok": True, "contradictions": answer["contradictions"], "attempts": attempts}
        retry_error = problem
    return {"ok": False, "contradictions": [], "attempts": attempts}


# ---------------------------------------------------------------- corruptions

def longest_word(text: str) -> str:
    words = [w for w in re.findall(r"[А-Яа-яЁёA-Za-z0-9]+", text or "") if len(w) > 4]
    return max(words, key=len) if words else ""


def corrupt_v2(result: dict[str, Any], kind: str):
    """(corrupted copy, description, field, marker, expected types) or None."""
    spoiled = copy.deepcopy(result)
    if kind == "replace":
        step = spoiled.get("next_step") or {}
        if step.get("action"):
            old = step["action"]
            step["action"] = "Отправить коммерческое предложение по электронной почте"
            spoiled["next_step"] = step
            return spoiled, f"next_step.action {old!r} replaced with a different action", \
                "next_step.action", "коммерческое предложение", ("выдумка", "искажение")
        if spoiled.get("client_request"):
            old = spoiled["client_request"]
            spoiled["client_request"] = "Клиент просил подобрать запчасти для грузового прицепа"
            return spoiled, f"client_request replaced (was {old[:40]!r})", \
                "client_request", "прицеп", ("выдумка", "искажение")
        return None
    if kind == "delete":
        if spoiled.get("objections"):
            gone = spoiled["objections"].pop(0)
            return spoiled, f"objection removed: {gone.get('objection', '')[:40]!r}", \
                "objections", longest_word(gone.get("objection", "")), ("пропуск",)
        if spoiled.get("amounts_mentioned"):
            gone = spoiled["amounts_mentioned"].pop(0)
            return spoiled, f"amount removed: {gone.get('value')}", \
                "amounts_mentioned", str(gone.get("value", "")).rstrip("0").rstrip("."), ("пропуск",)
        if spoiled.get("agreements"):
            gone = spoiled["agreements"].pop(0)
            return spoiled, f"agreement removed: {gone[:40]!r}", \
                "agreements", longest_word(gone), ("пропуск",)
        return None
    if kind == "distort":
        amounts = spoiled.get("amounts_mentioned") or []
        if amounts:
            old = amounts[0].get("value")
            amounts[0]["value"] = round(float(old) * 7 + 13, 2)
            return spoiled, f"amount {old} distorted into {amounts[0]['value']}", \
                "amounts_mentioned", str(amounts[0]["value"]).rstrip("0").rstrip("."), \
                ("искажение", "выдумка")
        people = spoiled.get("participants") or {}
        if people.get("client"):
            old = people["client"]
            people["client"] = "Вениамин Сидорчук"
            return spoiled, f"client name {old!r} distorted", \
                "participants.client", "Вениамин", ("искажение", "выдумка")
        step = spoiled.get("next_step") or {}
        if step.get("deadline"):
            old = step["deadline"]
            step["deadline"] = "29 февраля в 23:45"
            spoiled["next_step"] = step
            return spoiled, f"deadline {old!r} distorted", \
                "next_step.deadline", "29 февраля", ("искажение", "выдумка")
        return None
    raise ValueError(kind)


def caught_v2(found: list[dict[str, Any]], field: str, marker: str, types: tuple[str, ...]) -> bool:
    target = field.split(".")[-1].casefold()
    needle = (marker or "").casefold()
    for item in found:
        text = f"{item.get('field', '')} {item.get('detail', '')}".casefold()
        if needle and needle in text and item.get("type") in types:
            return True
        if target in text and item.get("type") in types:
            return True
    return False


# ------------------------------------------------------------------------ run

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="limit the number of calls (pilot)")
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--compare-v1", action="store_true",
                        help="repeat the five v1 control cases with the same corruptions")
    args = parser.parse_args()

    v1 = load_v1()
    cfg_mta = load_config()
    cfg = h.load_config()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with db.connect(cfg_mta.database_url) as conn:
        rows = [dict(r) for r in conn.execute(SAMPLE_SQL).fetchall()]
    random.seed(args.seed)
    sample = sorted(random.sample(rows, 10), key=lambda r: r["id"])
    if args.limit:
        sample = sample[: args.limit]

    report: dict[str, Any] = {
        "version": "v2", "seed": args.seed, "model": cfg.ollama_model,
        "candidates": len(rows), "sample_ids": [r["id"] for r in sample],
        "retries": RETRIES, "background": [], "control": [], "compare_v1": [],
    }
    unparsed = 0

    print(f"candidates {len(rows)}, sample {len(sample)}, seed {args.seed}, model {cfg.ollama_model}")
    print("\nbackground: the judge on untouched parses")
    for n, row in enumerate(sample, 1):
        res = ask(cfg, v1, row["transcript"], row["result"])
        unparsed += 0 if res["ok"] else 1
        report["background"].append({"id": row["id"], "ok": res["ok"],
                                     "contradictions": res["contradictions"],
                                     "attempts": len(res["attempts"])})
        print(f"  [{n}/{len(sample)}] id={row['id']:6d} findings {len(res['contradictions'])}"
              f" attempts {len(res['attempts'])}" + ("" if res["ok"] else "  UNPARSED"), flush=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\ncorruption control: 10 calls against three kinds")
    for n, row in enumerate(sample, 1):
        for kind in KINDS:
            made = corrupt_v2(row["result"], kind)
            if made is None:
                report["control"].append({"id": row["id"], "kind": kind, "applicable": False})
                print(f"  id={row['id']:6d} [{kind:7}] not applicable: nothing to corrupt")
                continue
            spoiled, what, field, marker, types = made
            res = ask(cfg, v1, row["transcript"], spoiled)
            unparsed += 0 if res["ok"] else 1
            hit = caught_v2(res["contradictions"], field, marker, types)
            report["control"].append({"id": row["id"], "kind": kind, "applicable": True,
                                      "injected": what, "field": field, "marker": marker,
                                      "expected_types": list(types), "caught": hit,
                                      "ok": res["ok"], "attempts": len(res["attempts"]),
                                      "contradictions": res["contradictions"]})
            print(f"  id={row['id']:6d} [{kind:7}] {what[:52]:52} -> "
                  f"{'CAUGHT' if hit else 'missed'}", flush=True)
            out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    if args.compare_v1 and V1_REPORT.exists():
        print("\ncomparison with v1 on its five control cases")
        old = json.loads(V1_REPORT.read_text(encoding="utf-8"))
        ids = [(c["id"], c["kind"]) for c in old.get("control", [])]
        with db.connect(cfg_mta.database_url) as conn:
            by_id = {r["id"]: dict(r) for r in conn.execute(
                "SELECT id, transcript, result FROM documents WHERE id = ANY(%s)",
                ([i for i, _ in ids],)).fetchall()}
        for doc_id, kind in ids:
            row = by_id.get(doc_id)
            if row is None:
                report["compare_v1"].append({"id": doc_id, "kind": kind, "found": False})
                continue
            spoiled, what, field, marker = v1.corrupt(row["result"], kind)
            res = ask(cfg, v1, row["transcript"], spoiled)
            unparsed += 0 if res["ok"] else 1
            hit = v1.caught(res["contradictions"], field, marker)
            report["compare_v1"].append({"id": doc_id, "kind": kind, "caught": hit,
                                         "ok": res["ok"], "attempts": len(res["attempts"])})
            print(f"  id={doc_id:6d} [{kind:9}] -> {'CAUGHT' if hit else 'missed'}", flush=True)
            out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    report["unparsed"] = unparsed
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nunparsed answers: {unparsed}\nresult: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
