"""The gate that must pass before the first run (DESIGN 10). The model is never called.

Comments and docstrings translated to English for review; logic unchanged.

Usage: python experiments/gate.py
Exit code 0 means the gate passed, 1 means some items are still open.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness as h  # noqa: E402
import freeze  # noqa: E402
from mta import pipeline as mta_pipeline  # noqa: E402
from mta import schema as schema_mod  # noqa: E402

SAFE_BUILTINS = {"all": all, "any": any, "len": len}
EXPECTED_SCORE = {"good": 10, "mid": 4}
FORBIDDEN_IN_HARNESS = ("psycopg", "out-json", "out_dir", "connect(", "from mta.db", "import db")

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def ev(expr: str, result: dict[str, Any], raw: str = "") -> bool:
    return bool(eval(expr, {"__builtins__": SAFE_BUILTINS}, {"r": result, "raw": raw}))


def load_reference(base: str) -> dict[str, Any]:
    with (h.SUITE_DIR / "bases" / base / "reference.json").open(encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------- freeze

def check_frozen(suite: dict[str, Any]) -> None:
    rows = freeze.hashes()
    recorded = freeze.read_recorded()
    if not recorded:
        check("10.2 freeze: FROZEN.md", False, "file missing or holds no hashes")
        return
    bad = [name for name, digest in rows if recorded.get(name) != digest]
    extra = sorted(set(recorded) - {name for name, _ in rows})
    check(
        "10.2 freeze: sha256 match FROZEN.md",
        not bad and not extra,
        "; ".join(bad + [f"extra {e}" for e in extra]) or f"files: {len(rows)}",
    )


def check_render(suite: dict[str, Any]) -> None:
    for base in suite["bases"]:
        path = h.SUITE_DIR / "bases" / base / "transcript.txt"
        rendered = h.render(h.load_base(base), suite["timing"]) + "\n"
        check(
            f"10.1 transcript {base} is reproduced by the renderer from utterances.json",
            path.read_text(encoding="utf-8") == rendered,
        )


# ----------------------------------------------------------------- references

def check_references(suite: dict[str, Any]) -> None:
    sales_schema = schema_mod.load_schema(str(h.SCHEMA_PATH))
    service = suite["service_fields"]
    for base in suite["bases"]:
        ref = load_reference(base)
        problem = schema_mod.validation_error(sales_schema, ref)
        check(f"10.3 reference {base} is valid against sales.json", problem is None, problem or "")

        computed = mta_pipeline.compute_score(ref["score_breakdown"], ref["contact_type"])
        check(
            f"10.6 reference {base}: compute_score = {EXPECTED_SCORE[base]} (table 4.3)",
            computed == ref["score"] == EXPECTED_SCORE[base],
            f"compute_score={computed}, reference says {ref['score']}",
        )
        check(
            f"10.6 reference {base}: contact_type is not the internal-call type",
            ref["contact_type"] != mta_pipeline.INTERNAL_CONTACT_TYPE,
            ref["contact_type"],
        )
        same_service = (
            ref["source_kind"] == service["source_kind"]
            and ref["employee_id"] == service["identity"]["employee"]
            and ref["identity"] == service["identity"]
            and ref["source_ref"] == {"file": service["file_name"], "timecodes": service["timecodes"]}
            and ref["contact_type"] == service["contact_type"]
        )
        check(f"10.5 reference {base}: service fields match cases.json", same_service)


# ---------------------------------------------------------------------- cases

def check_cases(suite: dict[str, Any]) -> None:
    for case in suite["cases"]:
        base = case["criterion_base"]
        anchors_ok = True
        for other in [b for b in suite["bases"] if case["id"] in suite["plan"][b]]:
            anchors = suite["anchors"][other]
            n = len(h.load_base(other))
            for kind in ("attack", "filler"):
                for ins in case["inserts"][kind]:
                    anchors_ok &= ins["anchor"] in anchors
                    anchors_ok &= 0 <= int(anchors.get(ins["anchor"], -1)) <= n
                    anchors_ok &= ins["role"] in h.merge.ROLE_LABELS
        check(f"case {case['id']}: insertion anchors and roles are valid on every base", anchors_ok)

        counts = {
            kind: [len(i["text"].split()) for i in case["inserts"][kind]]
            for kind in ("attack", "filler")
        }
        same_shape = len(counts["attack"]) == len(counts["filler"])
        check(
            f"case {case['id']}: filler of the same shape, words {counts['attack']} vs {counts['filler']}",
            same_shape,
        )

        ref = load_reference(base)
        check(
            f"case {case['id']}: precondition_clean holds on the reference {base}",
            ev(case["precondition_clean"], ref),
            case["precondition_clean"],
        )
        check(
            f"case {case['id']}: success criterion does NOT fire on the {base} reference",
            not ev(case["success"], ref, raw=""),
            case["success"],
        )

    for base, ids in suite["plan"].items():
        known = {c["id"] for c in suite["cases"]}
        check(f"run plan for {base}: all cases are known", set(ids) <= known,
              ", ".join(sorted(set(ids) - known)))
        check(f"run plan for {base}: the cases with criteria on this base are included",
              {c["id"] for c in suite["cases"] if c["criterion_base"] == base} <= set(ids))

    canary = suite["canary"]
    g = [c for c in suite["cases"] if c["id"] == "G"][0]
    check(
        "case G: the canary is in the payload and absent from the filler",
        all(canary in i["text"] for i in g["inserts"]["attack"])
        and all(canary not in i["text"] for i in g["inserts"]["filler"]),
    )
    for base in suite["bases"]:
        text = (h.SUITE_DIR / "bases" / base / "transcript.txt").read_text(encoding="utf-8")
        check(f"the canary does not occur in base {base}", canary not in text)


# ------------------------------------------------- the harness never touches the DB

def check_harness_isolation() -> None:
    source = (h.SUITE_DIR / "experiments" / "harness.py").read_text(encoding="utf-8")
    hits = [token for token in FORBIDDEN_IN_HARNESS if token in source]
    check("10.11 the harness does not reach the MTA database or write to out-json", not hits, ", ".join(hits))

    gitignore = (h.SUITE_DIR.parent / ".gitignore").read_text(encoding="utf-8").splitlines()
    ignored = {line.strip().rstrip("/") for line in gitignore}
    check("10.10 analytic/ and audio-meet/ are covered by .gitignore", {"analytic", "audio-meet"} <= ignored)


# ------------------------------------------------------ equivalence gate (6.2)

class FakeClient:
    """Stands in for `OllamaClient` on the production path: no network involved."""

    def __init__(self, answers: list[dict[str, Any]]) -> None:
        self.answers = answers
        self.calls: list[dict[str, Any]] = []

    def chat_json(self, *, system, user, schema, temperature=0.0, top_p=None):
        self.calls.append(
            {"system": system, "user": user, "schema": schema,
             "temperature": temperature, "top_p": top_p}
        )
        return copy.deepcopy(self.answers[min(len(self.calls) - 1, len(self.answers) - 1)])


def make_transport(answers: list[dict[str, Any]], sink: list[dict[str, Any]]):
    def transport(payload: dict[str, Any]) -> dict[str, Any]:
        sink.append(copy.deepcopy(payload))
        answer = answers[min(len(sink) - 1, len(answers) - 1)]
        return {
            "message": {"content": json.dumps(answer, ensure_ascii=False)},
            "prompt_eval_count": 111,
            "eval_count": 222,
        }
    return transport


def model_answer(base: str) -> dict[str, Any]:
    """The model answer: the reference minus the fields the model never returns."""
    answer = copy.deepcopy(load_reference(base))
    for field in schema_mod.MODEL_EXCLUDED_FIELDS:
        answer.pop(field, None)
    return answer


def run_pair(suite: dict[str, Any], answers: list[dict[str, Any]], body: str):
    service = suite["service_fields"]
    identity = h.service_identity(service)
    cfg_ns = SimpleNamespace(schema_path=h.SCHEMA_PATH, llm_retries=2)

    client = FakeClient(answers)
    prod = mta_pipeline._extract(
        cfg_ns, client,
        source_kind=service["source_kind"], identity=identity,
        file_name=service["file_name"], timecodes=service["timecodes"],
        contact_type=service["contact_type"], body=body,
    )

    cfg = h.HarnessConfig(
        schema_path=h.SCHEMA_PATH, llm_retries=2, ollama_url="stub://",
        ollama_model="stub-model", ollama_think=False, llm_timeout_s=1,
    )
    sink: list[dict[str, Any]] = []
    own, attempts = h.extract(
        cfg, service=service, body=body, transport=make_transport(answers, sink)
    )
    return prod, client, own, attempts, sink


def check_equivalence(suite: dict[str, Any]) -> None:
    base = "mid"
    body = (h.SUITE_DIR / "bases" / base / "transcript.txt").read_text(encoding="utf-8").strip()
    good = model_answer(base)
    broken = copy.deepcopy(good)
    broken.pop("summary")

    for label, answers, expect_attempts in (
        ("no retry", [good], 1),
        ("retry on schema failure", [broken, good], 2),
    ):
        prod, client, own, attempts, sink = run_pair(suite, answers, body)
        check(f"6.2 {label}: number of calls matches ({expect_attempts})",
              len(client.calls) == len(sink) == expect_attempts,
              f"production {len(client.calls)}, harness {len(sink)}")
        prompts_equal = all(
            client.calls[i]["system"] == attempts[i]["prompt_system"]
            and client.calls[i]["user"] == attempts[i]["prompt_user"]
            for i in range(min(len(client.calls), len(attempts)))
        )
        check(f"6.2 {label}: prompts match byte for byte", prompts_equal)
        schema_equal = all(
            client.calls[i]["schema"] == sink[i]["format"] for i in range(len(sink))
        )
        check(f"6.2 {label}: the trimmed schema sent in format matches", schema_equal)
        options_equal = all(
            sink[i]["options"] == {"temperature": client.calls[i]["temperature"],
                                   "top_p": client.calls[i]["top_p"]}
            for i in range(len(sink))
        )
        check(f"6.2 {label}: temperature and top_p match", options_equal)
        check(f"6.2 {label}: the resulting JSON matches on every field", prod == own,
              "" if prod == own else "objects differ")
        check(f"6.2 {label}: score and score_breakdown match",
              prod.get("score") == own.get("score")
              and prod.get("score_breakdown") == own.get("score_breakdown"),
              f"{prod.get('score')} vs {own.get('score')}")

    prod, client, own, attempts, sink = run_pair(suite, [broken, good], body)
    check("6.2 the second prompt carries the validator-error block",
          "ВНИМАНИЕ" in attempts[1]["prompt_user"]
          and attempts[1]["prompt_user"] != attempts[0]["prompt_user"])
    check("6.2 raw response and token counters are kept",
          attempts[1]["raw_response"] is not None
          and attempts[1]["tokens"] == {"prompt_eval_count": 111, "eval_count": 222})


def main() -> int:
    suite = h.load_suite()
    check_frozen(suite)
    check_render(suite)
    check_references(suite)
    check_cases(suite)
    check_harness_isolation()
    check_equivalence(suite)

    width = max(len(name) for name, _, _ in results)
    failed = 0
    for name, ok, detail in results:
        mark = "ok  " if ok else "FAILED"
        failed += 0 if ok else 1
        tail = f"  {detail}" if detail and not ok else ""
        print(f"{mark} {name.ljust(width)}{tail}")
    print(f"\nchecks: {len(results)}, failures: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
