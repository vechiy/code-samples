"""The human_audio condition: ASR and diarization once per file, then extraction.

Comments and docstrings translated to English for review; logic unchanged.

Usage (under an account that can read the MTA .env):
    python experiments/run_human_audio.py freeze     ASR, diarization, freezing
    python experiments/run_human_audio.py extract    extraction over frozen transcripts

The MTA database is never touched: only the pure ASR, diarization and utterance
assembly functions are used. Re-running ASR is forbidden: a file with a ready
artifact is skipped.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["MTA_HOME"] = os.environ.get("MTA_ROOT", "/opt/manager-talk-analyzer")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness as h  # noqa: E402

from mta import asr, diarize, merge  # noqa: E402
from mta.config import load_config  # noqa: E402
from mta.pipeline import build_reply_utterances  # noqa: E402

AUDIO_DIR = h.SUITE_DIR.parent / "audio-meet" / "mats-injection"
OUT = h.SUITE_DIR.parent / "analytic" / "mats-injection" / "audio"
FROZEN = OUT / "frozen"
MAP_PATH = h.SUITE_DIR / "scripts" / "audio_map.md"
CALL_DIRECTION = "in"


def audio_map() -> dict[str, str]:
    """{case: file name} from the name map. Rows without a working name are skipped."""
    pairs = {}
    for line in MAP_PATH.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\|\s*(clean|A|B|C|D|E|F2|G)\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|", line)
        if m:
            pairs[m.group(1)] = m.group(3)
    return pairs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_one(cfg, case: str, name: str) -> dict[str, Any]:
    """One ASR and diarization pass, roles from the first utterance, raw material frozen."""
    path = AUDIO_DIR / name
    if path.suffix.lower() not in asr_suffixes():
        raise SystemExit(f"{name}: this extension is not accepted by the MTA pipeline")
    diag: dict[str, Any] = {}
    with asr.ConvertedAudio(path) as wav:
        speech = asr.speech_mask(wav)
        segments = diarize.diarize(cfg, wav)
        if not segments:
            raise SystemExit(f"{name}: diarization returned no segments")
        words, duration = asr.transcribe(cfg, wav)
        if not words:
            raise SystemExit(f"{name}: ASR recognised no words at all")
        utterances = build_reply_utterances(
            cfg, words, segments, wav_path=wav, speech=speech, employee=None, diag=diag)
        words = diag.get("words_final", words)

    speakers = merge.speakers_of(utterances)
    # Roles without an LLM call and without a voice print: by the script the manager
    # opens the call, so the speaker of the first utterance is the manager and the
    # other party is the client.
    first = utterances[0].speaker
    roles = {sp: ("manager" if sp == first else "client") for sp in speakers}
    labels = merge.role_label_map(roles, speakers)
    transcript = merge.render_transcript(utterances, labels)
    left = merge.speakers_in_transcript(transcript)
    if left:
        raise SystemExit(f"{name}: raw speaker labels left in the transcript: {left}")

    return {
        "case": case, "file": name, "audio_sha256": sha256_file(path),
        "duration_s": duration, "speakers": speakers, "roles": roles,
        "labels": labels, "transcript": transcript,
        "words": [{"start": w.start, "end": w.end, "text": w.text} for w in words],
        "speaker_segments": [{"start": s.start, "end": s.end, "speaker": s.speaker}
                             for s in segments],
        "utterances": [{"speaker": u.speaker, "start": u.start, "end": u.end, "text": u.text}
                       for u in utterances],
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }


def asr_suffixes() -> tuple[str, ...]:
    from mta import files as files_mod
    return tuple(s.lower() for s in files_mod.AUDIO_SUFFIXES)


def cmd_freeze() -> int:
    cfg = load_config()
    FROZEN.mkdir(parents=True, exist_ok=True)
    pairs = audio_map()
    print(f"files in the map: {len(pairs)}")
    for case, name in pairs.items():
        out = FROZEN / f"{case}.json"
        if out.exists():
            print(f"{case}: already frozen, ASR is not repeated")
            continue
        data = freeze_one(cfg, case, name)
        out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{case}: words {len(data['words'])}, utterances {len(data['utterances'])}, "
              f"segments {len(data['speaker_segments'])}, {data['duration_s']:.1f} s")
    return 0


def load_frozen() -> dict[str, dict[str, Any]]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(FROZEN.glob("*.json"))}


def filler_transcript(suite: dict[str, Any], clean: dict[str, Any], case: dict[str, Any]) -> str:
    """The audio filler is substituted as text into the frozen clean transcript."""
    utterances = [{"role": clean["roles"][u["speaker"]], "text": u["text"]}
                  for u in clean["utterances"]]
    anchor = anchor_index(clean, case)
    inserts = [{"anchor": "audio", "role": i["role"], "text": i["text"]}
               for i in case["inserts"]["filler"]]
    merged = h.with_inserts(utterances, inserts, {"audio": anchor})
    return h.render(merged, suite["timing"])


def anchor_index(clean: dict[str, Any], case: dict[str, Any]) -> int:
    """Index of the clean utterance the insertion goes after.

    The recorded utterances follow the same order as the base, so the anchor index
    is taken from the base. The role is verified: if someone else speaks at that
    point, the recording has diverged from the base and blind insertion is unsafe.
    """
    anchor_name = case["inserts"]["filler"][0]["anchor"]
    index = {"middle_manager": 7, "after_client_ok": 10}[anchor_name]
    base = h.load_base("mid")
    if len(clean["utterances"]) != len(base):
        raise SystemExit(
            f"the clean recording has {len(clean['utterances'])} utterances, the base has"
            f" {len(base)}: the filler anchor cannot be taken by index")
    speaker = clean["utterances"][index - 1]["speaker"]
    if clean["roles"][speaker] != base[index - 1]["role"]:
        raise SystemExit(f"utterance {index} of the recording belongs to a different role than in the base")
    return index


def cmd_extract() -> int:
    suite = h.load_suite()
    cases = {c["id"]: c for c in suite["cases"]}
    cfg = h.load_config()
    frozen = load_frozen()
    if "clean" not in frozen:
        raise SystemExit("no frozen clean recording, run freeze first")
    OUT.mkdir(parents=True, exist_ok=True)
    sink_path = OUT / "results.jsonl"
    service_base = suite["service_fields"]
    plan: list[tuple[str, str, str]] = [("clean", "clean", "clean")]
    for case_id in sorted(k for k in frozen if k != "clean"):
        plan.append((case_id, case_id, "attack"))
        plan.append((case_id, case_id, "filler"))
    print(f"variants: {len(plan)}, repeats: {suite['repeats']}")

    with sink_path.open("a", encoding="utf-8") as sink:
        for case_id, _, kind in plan:
            if kind == "filler":
                body = filler_transcript(suite, frozen["clean"], cases[case_id])
                file_name = frozen["clean"]["file"]
                audio_sha = None
            else:
                body = frozen[case_id]["transcript"]
                file_name = frozen[case_id]["file"]
                audio_sha = frozen[case_id]["audio_sha256"]
            service = dict(service_base, file_name=file_name)
            for repeat in range(1, int(suite["repeats"]) + 1):
                run_id = f"{suite['suite']}.audio.{case_id}.{kind}.r{repeat}"
                started = datetime.now(timezone.utc).isoformat()
                result, attempts = h.extract(cfg, service=service, body=body)
                common = {"suite": suite["suite"], "base": "mid", "case_id": case_id,
                          "condition": "human_audio", "kind": kind, "repeat_no": repeat,
                          "run_id": run_id, "started_at": started,
                          "transcript_sha256": h.sha256_text(body), "audio_sha256": audio_sha}
                for record in attempts:
                    sink.write(json.dumps({"record": "attempt", **common, **record,
                                           "attempts_total": len(attempts)},
                                          ensure_ascii=False) + "\n")
                last = attempts[-1]
                sink.write(json.dumps({
                    "record": "run", **common, "attempts_total": len(attempts),
                    "attempt_used": len(attempts) if result is not None else None,
                    "result_json": result, "score": (result or {}).get("score"),
                    "raw_response": last.get("raw_response"),
                    "error": None if result is not None else last.get("error"),
                    "model": last.get("model"), "options": last.get("options"),
                    "schema_sha256": last.get("schema_sha256"), "tokens": last.get("tokens"),
                    "durations": last.get("durations"), "latency_ms": last.get("latency_ms"),
                }, ensure_ascii=False) + "\n")
                sink.flush()
                print(f"{run_id}  attempts {len(attempts)}  "
                      f"{'error' if result is None else 'score=' + str(result.get('score'))}")
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in {"freeze", "extract"}:
        raise SystemExit(__doc__)
    return cmd_freeze() if sys.argv[1] == "freeze" else cmd_extract()


if __name__ == "__main__":
    raise SystemExit(main())
