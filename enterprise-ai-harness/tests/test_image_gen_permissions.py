"""The image_gen permission, the generate_image tool gate and image serving (hermetic).

Comments and docstrings translated to English for review; logic unchanged.

No network, no DB: crypto.tokenize, image_client.generate_for_user and the usage
journal write are replaced with stubs, and files go into a temporary directory.

It checks that:
  - without the image_gen permission the tool is NOT REGISTERED in the schema
    handed to the model at all;
  - a runtime call without the permission is refused (the second half of the
    double gate);
  - the prompt block is injected only with the image_gen permission;
  - masking acts as a stop-validator: a substitution in the prompt, blocked=True or
    an unavailable service -> refusal, and we do NOT go outside (fail-closed);
  - the payload for the frontend: success -> viz_hint image, refusal -> no block at all;
  - resolve_image_path: a foreign owner, traversal and a bad extension -> None.

Assertions match Russian substrings of production messages, so those stay as they
are; their English meaning is given in an adjacent comment.

Run: python tests/test_image_gen_permissions.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import crypto  # noqa: E402
import db  # noqa: E402
import image_client  # noqa: E402
import usage_storage  # noqa: E402
import prompts  # noqa: E402
from routes import chats  # noqa: E402

GROUP = "image-gen-permissions-unit"
results: list[tuple[str, str, str]] = []


def rec(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, "PASS" if ok else "FAIL", detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def _names(permissions: dict | None) -> set[str]:
    return {tool["function"]["name"] for tool in db.tools_for(permissions)}


def _tokenize_result(masked: str, blocked: bool = False) -> crypto.TokenizeResult:
    return crypto.TokenizeResult(
        request_id="rid", session_id="sid", blocked=blocked, block_reasons=[],
        masked_prompt=masked, masked_attachments=[], detected_entities=[],
        processing_time_ms=1,
    )


class _Calls:
    """Counts calls to the external API — proof that fail-closed never goes outside."""

    def __init__(self) -> None:
        self.count = 0
        self.last_model: str | None = None

    def generate_for_user(
        self,
        prompt: str,
        user_id: int,
        model: str | None = None,
        reference_image_ids=None,
    ) -> dict:
        self.count += 1
        self.last_model = model
        refs = list(reference_image_ids or [])
        return {"image_id": f"{user_id}_" + "a" * 32 + ".webp",
                "model": model or "gpt-img-low",
                "elapsed_sec": 1.0, "content_id": "cid",
                "img2img": bool(refs),
                "reference_image_ids": refs,
                "reference_image_id": refs[0] if refs else None,
                "reference_count": len(refs)}


def _run_tool(prompt: str, tokenize_stub, calls: _Calls) -> dict:
    """Call the tool with crypto.tokenize and image_client.generate_for_user stubbed."""
    original_tokenize = crypto.tokenize
    original_generate = image_client.generate_for_user
    # The tool writes a row into the usage journal — the stub is mandatory, or a
    # hermetic test would pour made-up costs into the live dev database.
    original_log = usage_storage.log_image_usage
    crypto.tokenize = tokenize_stub
    image_client.generate_for_user = calls.generate_for_user
    usage_storage.log_image_usage = lambda **kwargs: None
    try:
        return db.tool_generate_image(prompt, permissions={"image_gen": True}, user_id=7)
    finally:
        crypto.tokenize = original_tokenize
        image_client.generate_for_user = original_generate
        usage_storage.log_image_usage = original_log


def test_not_registered_without_permission() -> None:
    without = _names({})
    without_none = _names(None)
    with_right = _names({"image_gen": True})
    ok = ("generate_image" not in without and "generate_image" not in without_none
          and "generate_image" in with_right)
    rec("generate_image: not registered without the permission, present with it", ok,
        f"no permissions={'generate_image' in without}, None={'generate_image' in without_none}, "
        f"with image_gen={'generate_image' in with_right}")


def test_runtime_gate() -> None:
    empty = db.tool_generate_image("cat", permissions={}, user_id=7)
    none = db.tool_generate_image("cat", permissions=None, user_id=7)
    other = db.tool_generate_image("cat", permissions={"design": True}, user_id=7)
    # "прав" is the stem of "permissions" in the production refusal message.
    ok = all("error" in r and "прав" in r["error"].lower() for r in (empty, none, other))
    rec("runtime call without the permission is refused (fail-closed, incl. permissions=None)", ok,
        f"no permissions={empty.get('error')}, design={other.get('error')}")


def test_dispatch_gate() -> None:
    denied = db.dispatch_tool("generate_image", {"prompt": "cat"}, permissions={}, user_id=7)
    ok = "error" in denied and "прав" in denied["error"].lower()
    rec("dispatch_tool refuses generate_image without the permission", ok,
        denied.get("error", ""))


def test_prompt_gate() -> None:
    base = prompts.build_system_prompt()
    with_right = prompts.build_system_prompt(image_gen=True)
    others = prompts.build_system_prompt(olap=True, websearch=True, rag_read=True,
                                         file_read=True)
    ok = (prompts.IMAGE_GEN_PROMPT not in base
          and prompts.IMAGE_GEN_PROMPT in with_right
          and prompts.IMAGE_GEN_PROMPT not in others)
    rec("the prompt block is injected only with the image_gen permission", ok,
        f"no permissions={prompts.IMAGE_GEN_PROMPT in base}, "
        f"with permission={prompts.IMAGE_GEN_PROMPT in with_right}")


def test_masking_blocks_when_prompt_changed() -> None:
    calls = _Calls()
    # "ООО Ромашка" is a fictional company name ("Romashka LLC"), used here so the
    # masking stub has something to substitute.
    result = _run_tool("cat of ООО Ромашка",
                       lambda **kw: _tokenize_result("cat of TOKEN_1"), calls)
    ok = "error" in result and calls.count == 0
    rec("masking changed the text -> refusal, no call to the external API", ok,
        f"error={result.get('error')}, API calls={calls.count}")


def test_masking_blocked_flag() -> None:
    calls = _Calls()
    result = _run_tool("cat", lambda **kw: _tokenize_result("cat", blocked=True), calls)
    ok = "error" in result and calls.count == 0
    rec("blocked=True -> refusal, no call to the external API", ok,
        f"error={result.get('error')}, API calls={calls.count}")


def test_masking_service_down() -> None:
    calls = _Calls()

    def boom(**kwargs):
        raise crypto.CryptoError("network error")

    result = _run_tool("cat", boom, calls)
    ok = "error" in result and calls.count == 0
    rec("masking service unavailable -> refusal (fail-closed)", ok,
        f"error={result.get('error')}, API calls={calls.count}")


def test_clean_prompt_passes_unchanged() -> None:
    calls = _Calls()
    seen: dict = {}

    def passthrough(**kwargs):
        seen.update(kwargs)
        return _tokenize_result(kwargs["prompt"])

    result = _run_tool("a cat brewing shampoo", passthrough, calls)
    ok = (result.get("status") == "ok" and result.get("image_id")
          and calls.count == 1 and "url" not in result
          and seen.get("prompt") == "a cat brewing shampoo")
    rec("a clean prompt goes through: one generation, no provider URL handed out", ok,
        f"result={result}, API calls={calls.count}")


def test_empty_prompt() -> None:
    calls = _Calls()
    result = _run_tool("   ", lambda **kw: _tokenize_result("   "), calls)
    ok = "error" in result and calls.count == 0
    rec("an empty description -> refusal without touching the API", ok,
        result.get("error", ""))


def test_result_payload() -> None:
    ok_payload = chats._infer_result_payload(  # noqa: SLF001
        "generate_image",
        {"type": "generated_image", "status": "ok", "image_id": "7_" + "b" * 32 + ".webp",
         "model": "flux-1-schnell"},
    )
    denied_payload = chats._infer_result_payload(  # noqa: SLF001
        # "Insufficient permissions to generate images."
        "generate_image", {"error": "Недостаточно прав для генерации изображений."})
    ok = (ok_payload is not None
          and ok_payload["viz_hint"]["type"] == "image"
          and ok_payload["data"]["image_id"] == "7_" + "b" * 32 + ".webp"
          and "prompt" not in ok_payload["data"]
          and denied_payload is None)
    rec("payload: success -> viz_hint image without the prompt, refusal -> no block", ok,
        f"success={ok_payload}, refusal={denied_payload}")


def test_resolve_image_path() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        original_dir = image_client.GENERATED_DIR
        image_client.GENERATED_DIR = Path(tmp)
        try:
            name = "7_" + "c" * 32 + ".webp"
            (Path(tmp) / name).write_bytes(b"RIFF0000WEBP")
            (Path(tmp) / ("9_" + "d" * 32 + ".webp")).write_bytes(b"RIFF0000WEBP")
            outside = Path(tmp).parent / "secret.webp"
            outside.write_bytes(b"x")
            try:
                own = image_client.resolve_image_path(name, 7)
                alien = image_client.resolve_image_path("9_" + "d" * 32 + ".webp", 7)
                missing = image_client.resolve_image_path("7_" + "e" * 32 + ".webp", 7)
                traversal = image_client.resolve_image_path("../secret.webp", 7)
                sneaky = image_client.resolve_image_path("7_" + "c" * 32 + ".webp/../../secret.webp", 7)
                bad_ext = image_client.resolve_image_path("7_" + "c" * 32 + ".php", 7)
                empty = image_client.resolve_image_path("", 7)
            finally:
                outside.unlink(missing_ok=True)
            ok = (own is not None and own.name == name
                  and alien is None and missing is None and traversal is None
                  and sneaky is None and bad_ext is None and empty is None)
            rec("resolve_image_path: own file served, foreign/traversal/bad — None", ok,
                f"own={own}, foreign={alien}, missing={missing}, traversal={traversal}, "
                f"concatenated={sneaky}, extension={bad_ext}, empty={empty}")
        finally:
            image_client.GENERATED_DIR = original_dir


def test_permission_in_sources_of_truth() -> None:
    import auth
    ok = "image_gen" in auth.PERMISSIONS
    rec("image_gen is in auth.PERMISSIONS (the source of truth for the columns)", ok,
        str(auth.PERMISSIONS))


TESTS = [
    test_not_registered_without_permission,
    test_runtime_gate,
    test_dispatch_gate,
    test_prompt_gate,
    test_masking_blocks_when_prompt_changed,
    test_masking_blocked_flag,
    test_masking_service_down,
    test_clean_prompt_passes_unchanged,
    test_empty_prompt,
    test_result_payload,
    test_resolve_image_path,
    test_permission_in_sources_of_truth,
]


def run() -> list[tuple[str, str, str]]:
    results.clear()
    for test in TESTS:
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            rec(test.__name__, False, f"exception {type(exc).__name__}: {exc}")
    return list(results)


if __name__ == "__main__":
    run()
    failed = [r for r in results if r[1] == "FAIL"]
    print(f"\n{GROUP}: {len(results) - len(failed)}/{len(results)} PASS")
    raise SystemExit(1 if failed else 0)
