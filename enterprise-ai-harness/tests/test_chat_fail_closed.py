"""Fail-closed masking in the chat: if the service fails, nothing leaves (hermetic).

Comments and docstrings translated to English for review; logic unchanged.

No network, no DB: crypto.tokenize is replaced with a stub that raises CryptoError,
while llm._post_to_llm and requests.post are replaced with spies. A real HTTP call
to the provider is impossible here: the requests.post spy raises AssertionError.

It checks that, with the masking service unavailable:
  - the exception propagates out instead of being swallowed inside the tool loop;
  - llm._post_to_llm is never called (the barrier sits BEFORE the provider call);
  - requests.post is never called (not one byte leaves the perimeter);
  - the user receives no chunk event at all.

Run: backend/venv/bin/python backend/tests/test_chat_fail_closed.py
"""

from __future__ import annotations

import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import crypto  # noqa: E402
import llm  # noqa: E402

GROUP = "chat-fail-closed-unit"
results: list[tuple[str, str, str]] = []


def rec(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, "PASS" if ok else "FAIL", detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def _model() -> llm.ModelInfo:
    """A cloud model: backend != ollama, so masking is mandatory."""
    return llm.ModelInfo(id="test-model", label="Test", provider="Test", backend="chadgpt")


class _Calls:
    """Counts calls to the provider — proof that fail-closed never goes outside."""

    def __init__(self) -> None:
        self.post_to_llm = 0
        self.requests_post = 0


def _run_with_broken_masking(calls: _Calls) -> tuple[BaseException | None, list[str]]:
    """One chat run with masking broken. Returns (exception, event names)."""
    events: list[str] = []

    original_tokenize = crypto.tokenize
    original_post_to_llm = llm._post_to_llm  # noqa: SLF001
    original_requests_post = llm.requests.post

    def broken_tokenize(**kwargs: Any) -> crypto.TokenizeResult:
        raise crypto.CryptoError("network error")

    def spy_post_to_llm(payload: dict, backend: str) -> dict:
        calls.post_to_llm += 1
        return {"choices": [{"message": {"content": "this must not happen"}}]}

    def spy_requests_post(*args: Any, **kwargs: Any) -> Any:
        calls.requests_post += 1
        raise AssertionError("a hermetic test tried to reach the network")

    crypto.tokenize = broken_tokenize
    llm._post_to_llm = spy_post_to_llm  # noqa: SLF001
    llm.requests.post = spy_requests_post
    try:
        llm.chat_with_tools_streaming(
            # "How much did we sell in July?" — an ordinary Russian user message.
            [{"role": "user", "content": "Сколько продали в июле?"}],
            model=_model(),
            on_event=lambda name, data: events.append(name),
        )
        return None, events
    except BaseException as exc:  # noqa: BLE001 — the exception is what we test for
        return exc, events
    finally:
        crypto.tokenize = original_tokenize
        llm._post_to_llm = original_post_to_llm  # noqa: SLF001
        llm.requests.post = original_requests_post


def test_masking_failure_raises() -> None:
    calls = _Calls()
    exc, _ = _run_with_broken_masking(calls)
    ok = isinstance(exc, RuntimeError)
    rec("masking failure -> exception out, not a silent answer", ok,
        f"{type(exc).__name__}: {exc}" if exc else "no exception was raised")


def test_provider_not_called() -> None:
    calls = _Calls()
    _run_with_broken_masking(calls)
    ok = calls.post_to_llm == 0
    rec("_post_to_llm never called (the barrier precedes the provider call)", ok,
        f"_post_to_llm calls={calls.post_to_llm}")


def test_nothing_left_the_perimeter() -> None:
    calls = _Calls()
    _run_with_broken_masking(calls)
    ok = calls.requests_post == 0
    rec("requests.post never called (nothing left the perimeter)", ok,
        f"requests.post calls={calls.requests_post}")


def test_no_chunks_emitted() -> None:
    calls = _Calls()
    _, events = _run_with_broken_masking(calls)
    ok = "chunk" not in events
    rec("no chunk event was handed to the user", ok,
        f"events={events or 'none'}")


TESTS = [
    test_masking_failure_raises,
    test_provider_not_called,
    test_nothing_left_the_perimeter,
    test_no_chunks_emitted,
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
