"""Running the text_injection condition: sequential, clean before the attacks (DESIGN 10.13).

Comments and docstrings translated to English for review; logic unchanged.

Usage:
    OLLAMA_URL=... OLLAMA_MODEL=... python experiments/run_text_injection.py [--phase clean|payload|all] [--dry-run]

Writes `analytic/mats-injection/<suite>/results.jsonl`: one line per call attempt
plus a summary line per logical run (DESIGN 6.1).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness as h  # noqa: E402

OUT_DIR = h.SUITE_DIR.parent / "analytic" / "mats-injection"


def variants(suite: dict[str, Any], phase: str) -> Iterator[tuple[str, str, str]]:
    """(base, case, kind). All of clean first, then attacks and fillers."""
    if phase in ("clean", "all"):
        for base in suite["bases"]:
            yield base, "clean", "clean"
    if phase in ("payload", "all"):
        for base in suite["bases"]:
            for case_id in suite["plan"][base]:
                for kind in ("attack", "filler"):
                    yield base, case_id, kind


def main() -> int:
    argv = sys.argv[1:]
    phase = "all"
    if "--phase" in argv:
        phase = argv[argv.index("--phase") + 1]
    dry_run = "--dry-run" in argv

    suite = h.load_suite()
    cases = {c["id"]: c for c in suite["cases"]}
    service = suite["service_fields"]
    repeats = int(suite["repeats"])
    plan = list(variants(suite, phase))
    print(f"variants: {len(plan)}, repeats: {repeats}, logical runs: {len(plan) * repeats}")
    if dry_run:
        for base, case_id, kind in plan:
            case = cases.get(case_id)
            body = h.variant_transcript(suite, base, case, kind)
            print(f"{base:5} {case_id:5} {kind:6} lines {len(body.splitlines()):3} "
                  f"sha256 {h.sha256_text(body)[:12]}")
        return 0

    cfg = h.load_config()
    out_path = OUT_DIR / suite["suite"] / "results.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"model {cfg.ollama_model}, retries {cfg.llm_retries}, output {out_path}")

    with out_path.open("a", encoding="utf-8") as sink:
        for base, case_id, kind in plan:
            case = cases.get(case_id)
            body = h.variant_transcript(suite, base, case, kind)
            for repeat in range(1, repeats + 1):
                run_id = f"{suite['suite']}.{base}.{case_id}.{kind}.r{repeat}"
                started = datetime.now(timezone.utc).isoformat()
                result, attempts = h.extract(cfg, service=service, body=body)
                common = {
                    "suite": suite["suite"], "base": base, "case_id": case_id,
                    "condition": suite["condition"], "kind": kind, "repeat_no": repeat,
                    "run_id": run_id, "started_at": started,
                    "transcript_sha256": h.sha256_text(body), "audio_sha256": None,
                }
                for record in attempts:
                    sink.write(json.dumps({"record": "attempt", **common, **record,
                                           "attempts_total": len(attempts)},
                                          ensure_ascii=False) + "\n")
                last = attempts[-1]
                sink.write(json.dumps({
                    "record": "run", **common,
                    "attempts_total": len(attempts),
                    "attempt_used": len(attempts) if result is not None else None,
                    "result_json": result,
                    "score": (result or {}).get("score"),
                    "raw_response": last.get("raw_response"),
                    "error": None if result is not None else last.get("error"),
                    "model": last.get("model"), "options": last.get("options"),
                    "schema_sha256": last.get("schema_sha256"),
                    "tokens": last.get("tokens"), "durations": last.get("durations"),
                    "latency_ms": last.get("latency_ms"),
                }, ensure_ascii=False) + "\n")
                sink.flush()
                status = "error" if result is None else f"score={result.get('score')}"
                print(f"{run_id}  attempts {len(attempts)}  {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
