"""Право image_gen, гейт тула generate_image и отдача картинок (герметично).

Ни сети, ни БД: crypto.tokenize, image_client.generate_for_user и запись
в журнал расхода подменяются
заглушками, файлы кладутся во временный каталог.

Проверяет:
  - без права image_gen тул не зарегистрирован в схеме для модели ВООБЩЕ;
  - рантайм-вызов без права отклоняется (вторая половина двойного гейта);
  - блок промпта подставляется только при праве image_gen;
  - маскирование работает как валидатор-стоп: замена в промпте, blocked=True или
    недоступность сервиса -> отказ, наружу НЕ ходим (fail-closed);
  - payload для фронта: успех -> viz_hint image, отказ -> блока нет вовсе;
  - resolve_image_path: чужой владелец, traversal и битый формат -> None.

Запуск: python tests/test_image_gen_permissions.py
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
    """Счётчик обращений к внешнему API — проверяем, что fail-closed не ходит наружу."""

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
    """Вызов тула с подменёнными crypto.tokenize и image_client.generate_for_user."""
    original_tokenize = crypto.tokenize
    original_generate = image_client.generate_for_user
    # Тул пишет строку в журнал расхода — заглушка обязательна, иначе
    # герметичный тест наливает выдуманные затраты в живую dev-базу.
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
    rec("generate_image: без права не зарегистрирован, с правом — есть", ok,
        f"без прав={'generate_image' in without}, None={'generate_image' in without_none}, "
        f"с image_gen={'generate_image' in with_right}")


def test_runtime_gate() -> None:
    empty = db.tool_generate_image("cat", permissions={}, user_id=7)
    none = db.tool_generate_image("cat", permissions=None, user_id=7)
    other = db.tool_generate_image("cat", permissions={"design": True}, user_id=7)
    ok = all("error" in r and "прав" in r["error"].lower() for r in (empty, none, other))
    rec("рантайм-вызов без права отклонён (fail-closed, в т.ч. permissions=None)", ok,
        f"без прав={empty.get('error')}, design={other.get('error')}")


def test_dispatch_gate() -> None:
    denied = db.dispatch_tool("generate_image", {"prompt": "cat"}, permissions={}, user_id=7)
    ok = "error" in denied and "прав" in denied["error"].lower()
    rec("dispatch_tool без права отклоняет generate_image", ok, denied.get("error", ""))


def test_prompt_gate() -> None:
    base = prompts.build_system_prompt()
    with_right = prompts.build_system_prompt(image_gen=True)
    others = prompts.build_system_prompt(olap=True, websearch=True, rag_read=True,
                                         file_read=True)
    ok = (prompts.IMAGE_GEN_PROMPT not in base
          and prompts.IMAGE_GEN_PROMPT in with_right
          and prompts.IMAGE_GEN_PROMPT not in others)
    rec("блок промпта подставляется только при праве image_gen", ok,
        f"без прав={prompts.IMAGE_GEN_PROMPT in base}, с правом={prompts.IMAGE_GEN_PROMPT in with_right}")


def test_masking_blocks_when_prompt_changed() -> None:
    calls = _Calls()
    result = _run_tool("cat of ООО Ромашка",
                       lambda **kw: _tokenize_result("cat of TOKEN_1"), calls)
    ok = "error" in result and calls.count == 0
    rec("маскирование заменило текст -> отказ, во внешний API не ходим", ok,
        f"error={result.get('error')}, вызовов API={calls.count}")


def test_masking_blocked_flag() -> None:
    calls = _Calls()
    result = _run_tool("cat", lambda **kw: _tokenize_result("cat", blocked=True), calls)
    ok = "error" in result and calls.count == 0
    rec("blocked=True -> отказ, во внешний API не ходим", ok,
        f"error={result.get('error')}, вызовов API={calls.count}")


def test_masking_service_down() -> None:
    calls = _Calls()

    def boom(**kwargs):
        raise crypto.CryptoError("network error")

    result = _run_tool("cat", boom, calls)
    ok = "error" in result and calls.count == 0
    rec("сервис маскирования недоступен -> отказ (fail-closed)", ok,
        f"error={result.get('error')}, вызовов API={calls.count}")


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
    rec("чистый промпт проходит: одна генерация, ссылки модели не отдаём", ok,
        f"result={result}, вызовов API={calls.count}")


def test_empty_prompt() -> None:
    calls = _Calls()
    result = _run_tool("   ", lambda **kw: _tokenize_result("   "), calls)
    ok = "error" in result and calls.count == 0
    rec("пустое описание -> отказ без обращения к API", ok, result.get("error", ""))


def test_result_payload() -> None:
    ok_payload = chats._infer_result_payload(  # noqa: SLF001
        "generate_image",
        {"type": "generated_image", "status": "ok", "image_id": "7_" + "b" * 32 + ".webp",
         "model": "flux-1-schnell"},
    )
    denied_payload = chats._infer_result_payload(  # noqa: SLF001
        "generate_image", {"error": "Недостаточно прав для генерации изображений."})
    ok = (ok_payload is not None
          and ok_payload["viz_hint"]["type"] == "image"
          and ok_payload["data"]["image_id"] == "7_" + "b" * 32 + ".webp"
          and "prompt" not in ok_payload["data"]
          and denied_payload is None)
    rec("payload: успех -> viz_hint image без промпта, отказ -> блока нет", ok,
        f"успех={ok_payload}, отказ={denied_payload}")


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
            rec("resolve_image_path: свой файл отдаётся, чужой/traversal/битый — None", ok,
                f"свой={own}, чужой={alien}, нет файла={missing}, traversal={traversal}, "
                f"склейка={sneaky}, расширение={bad_ext}, пусто={empty}")
        finally:
            image_client.GENERATED_DIR = original_dir


def test_permission_in_sources_of_truth() -> None:
    import auth
    ok = "image_gen" in auth.PERMISSIONS
    rec("image_gen в auth.PERMISSIONS (источник истины для колонок)", ok,
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
            rec(test.__name__, False, f"исключение {type(exc).__name__}: {exc}")
    return list(results)


if __name__ == "__main__":
    run()
    failed = [r for r in results if r[1] == "FAIL"]
    print(f"\n{GROUP}: {len(results) - len(failed)}/{len(results)} PASS")
    raise SystemExit(1 if failed else 0)
