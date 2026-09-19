"""Harness for the prompt-injection experiment on MTA extraction (DESIGN 6.2).

Comments and docstrings translated to English for review; logic unchanged.

Only two things here are ours: a thin Ollama client and the retry loop. Both
repeat `ollama.chat_json` and `pipeline._extract` literally, but keep the raw
response and the token counters that the production client throws away. Prompts,
the trimmed schema, `_force_service_fields` and `compute_score` are imported from
MTA rather than reimplemented.

MTA is opened read-only: no bytecode is written, the database is never touched,
and `MTA_HOME` is moved away from the MTA root so that `mta.config` does not read
its `.env`.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

sys.dont_write_bytecode = True

MTA_ROOT = Path(os.environ.get("MTA_ROOT", "/opt/manager-talk-analyzer"))
SUITE_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MTA_HOME", str(Path(__file__).resolve().parent))
if str(MTA_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(MTA_ROOT / "src"))

import httpx  # noqa: E402

from mta import merge, prompts  # noqa: E402
from mta import schema as schema_mod  # noqa: E402
from mta.ollama import OllamaError, _extract_json  # noqa: E402
from mta.pipeline import (  # noqa: E402
    EXTRACTION_TEMPERATURE,
    EXTRACTION_TOP_P,
    Identity,
    _force_service_fields,
)

SCHEMA_PATH = MTA_ROOT / "schemas" / "sales.json"


@dataclass(frozen=True)
class HarnessConfig:
    """Exactly the fields `pipeline._extract` reads, plus the client parameters."""

    schema_path: Path
    llm_retries: int
    ollama_url: str
    ollama_model: str
    ollama_think: bool
    llm_timeout_s: int


def load_config() -> HarnessConfig:
    """Endpoint and model come from the environment: they are not in the code."""
    url = os.environ.get("OLLAMA_URL", "").strip().rstrip("/")
    if not url or "<" in url:
        raise SystemExit(
            "OLLAMA_URL is not set. The values live in the MTA .env, which this account"
            " cannot read: pass OLLAMA_URL and OLLAMA_MODEL in the run environment."
        )
    model = os.environ.get("OLLAMA_MODEL", "").strip()
    if not model:
        raise SystemExit("OLLAMA_MODEL is not set: it must match the production MTA model.")
    return HarnessConfig(
        schema_path=SCHEMA_PATH,
        llm_retries=int(os.environ.get("LLM_RETRIES", "2")),
        ollama_url=url,
        ollama_model=model,
        ollama_think=os.environ.get("OLLAMA_THINK", "").strip().lower() in {"1", "true", "yes", "on"},
        llm_timeout_s=int(os.environ.get("LLM_TIMEOUT_S", "300")),
    )


# --------------------------------------------------------------------- suite

def load_suite() -> dict[str, Any]:
    with (SUITE_DIR / "cases" / "cases.json").open(encoding="utf-8") as fh:
        return json.load(fh)


def load_base(base: str) -> list[dict[str, str]]:
    with (SUITE_DIR / "bases" / base / "utterances.json").open(encoding="utf-8") as fh:
        return json.load(fh)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ transcript

def _duration_s(text: str, timing: dict[str, float]) -> float:
    words = len(text.split())
    return max(timing["min_utterance_s"], round(words / timing["words_per_second"]))


def render(utterances: list[dict[str, str]], timing: dict[str, float]) -> str:
    """The `merge.render_transcript` format: [ROLE | mm:ss-mm:ss]: text."""
    lines: list[str] = []
    clock = 0.0
    for utt in utterances:
        start = clock
        end = start + _duration_s(utt["text"], timing)
        label = merge.ROLE_LABELS[utt["role"]]
        lines.append(f"[{label} | {merge._fmt_ts(start)}-{merge._fmt_ts(end)}]: {utt['text']}")
        clock = end + timing["gap_s"]
    return "\n".join(lines)


def with_inserts(
    base_utterances: list[dict[str, str]],
    inserts: list[dict[str, Any]],
    anchors: dict[str, int],
) -> list[dict[str, str]]:
    """Insert after the utterance the anchor points at (1-based; 0 means the very start).

    Anchors are per base: the same point of the conversation sits at different
    numbers in different bases. Timecodes of later utterances shift by the length
    of the inserted line: a transcript with an extra utterance is longer, and the
    same happens in the recording.
    """
    out: list[dict[str, str]] = []
    by_anchor: dict[int, list[dict[str, Any]]] = {}
    for ins in inserts:
        by_anchor.setdefault(int(anchors[ins["anchor"]]), []).append(ins)
    for ins in by_anchor.get(0, []):
        out.append({"role": ins["role"], "text": ins["text"]})
    for index, utt in enumerate(base_utterances, start=1):
        out.append(utt)
        for ins in by_anchor.get(index, []):
            out.append({"role": ins["role"], "text": ins["text"]})
    return out


def variant_transcript(suite: dict[str, Any], base: str, case: dict[str, Any] | None, kind: str) -> str:
    """kind: clean | attack | filler."""
    utterances = load_base(base)
    if kind != "clean":
        utterances = with_inserts(utterances, case["inserts"][kind], suite["anchors"][base])
    return render(utterances, suite["timing"])


# ------------------------------------------------------------------ extraction

def trimmed_schema() -> dict[str, Any]:
    """The same trimmed schema the production path sends in `format`."""
    sales_schema = schema_mod.load_schema(str(SCHEMA_PATH))
    return schema_mod.without_fields(
        schema_mod.as_ollama_format(sales_schema), schema_mod.MODEL_EXCLUDED_FIELDS
    )


def service_identity(service: dict[str, Any]) -> Identity:
    ident = service["identity"]
    return Identity(ident["employee"], ident["method"], ident["similarity"])


def chat_raw(cfg: HarnessConfig, *, system: str, user: str, fmt: dict[str, Any],
             transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
             options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Request body literally as in `ollama.chat_json:158-171`, full response back.

    `options` is passed only by callers with their own call parameters (the
    contradiction judge runs at temperature 0 and without top_p). The default is
    the extraction parameters, as in production.
    """
    payload: dict[str, Any] = {
        "model": cfg.ollama_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "format": fmt,
        "stream": False,
        "think": cfg.ollama_think,
        "options": options if options is not None
        else {"temperature": EXTRACTION_TEMPERATURE, "top_p": EXTRACTION_TOP_P},
    }
    if transport is not None:
        return transport(payload)
    try:
        with httpx.Client(timeout=cfg.llm_timeout_s) as client:
            resp = client.post(f"{cfg.ollama_url}/api/chat", json=payload)
            if resp.status_code == 400 and "think" in resp.text.lower():
                payload.pop("think", None)
                resp = client.post(f"{cfg.ollama_url}/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as exc:
        raise OllamaError(f"request to Ollama failed: {exc}") from exc


def extract(
    cfg: HarnessConfig,
    *,
    service: dict[str, Any],
    body: str,
    transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Retry loop mirroring `pipeline._extract:156-222`. Returns (result, attempt records)."""
    sales_schema = schema_mod.load_schema(str(cfg.schema_path))
    fmt = schema_mod.without_fields(
        schema_mod.as_ollama_format(sales_schema), schema_mod.MODEL_EXCLUDED_FIELDS
    )
    identity = service_identity(service)
    attempts: list[dict[str, Any]] = []
    retry_error: str | None = None
    last_problem = "unknown error"

    for attempt in range(cfg.llm_retries + 1):
        user = prompts.extraction_user(
            source_kind=service["source_kind"],
            employee_id=identity.employee,
            file_name=service["file_name"],
            timecodes=service["timecodes"],
            contact_type=service["contact_type"],
            body=body,
            schema=fmt,
            retry_error=retry_error,
        )
        record: dict[str, Any] = {
            "attempt_no": attempt + 1,
            "prompt_system": prompts.EXTRACTION_SYSTEM,
            "prompt_user": user,
            "schema_sha256": sha256_text(json.dumps(fmt, ensure_ascii=False, sort_keys=True)),
            "model": cfg.ollama_model,
            "options": {
                "temperature": EXTRACTION_TEMPERATURE,
                "top_p": EXTRACTION_TOP_P,
                "think": cfg.ollama_think,
            },
            "raw_response": None,
            "parsed_ok": False,
            "validation_error": None,
            "result_json": None,
            "score": None,
            "latency_ms": None,
            "tokens": None,
            "durations": None,
            "error": None,
        }
        started = time.monotonic()
        try:
            data = chat_raw(cfg, system=prompts.EXTRACTION_SYSTEM, user=user, fmt=fmt,
                            transport=transport)
            record["latency_ms"] = int((time.monotonic() - started) * 1000)
            content = (data.get("message") or {}).get("content", "")
            record["raw_response"] = content
            record["tokens"] = {
                "prompt_eval_count": data.get("prompt_eval_count"),
                "eval_count": data.get("eval_count"),
            }
            record["durations"] = {
                "total_duration": data.get("total_duration"),
                "load_duration": data.get("load_duration"),
                "eval_duration": data.get("eval_duration"),
            }
            if not content.strip():
                raise OllamaError("Ollama returned empty content")
            result = _extract_json(content)
        except OllamaError as exc:
            record["latency_ms"] = record["latency_ms"] or int((time.monotonic() - started) * 1000)
            record["error"] = str(exc)
            last_problem = str(exc)
            retry_error = str(exc)
            attempts.append(record)
            continue

        record["parsed_ok"] = True
        result = _force_service_fields(
            result,
            source_kind=service["source_kind"],
            identity=identity,
            file_name=service["file_name"],
            timecodes=service["timecodes"],
            contact_type=service["contact_type"],
        )
        record["result_json"] = result
        record["score"] = result.get("score")

        problem = schema_mod.validation_error(sales_schema, result)
        record["validation_error"] = problem
        attempts.append(record)
        if problem is None:
            return result, attempts
        last_problem = problem
        retry_error = problem

    attempts[-1]["error"] = (
        f"the LLM did not return schema-valid JSON in {cfg.llm_retries + 1} attempts: {last_problem}"
    )
    return None, attempts
