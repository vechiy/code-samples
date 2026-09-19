"""Permissions on both OLAP tools and the prompt block (hermetic: no network, no DB).

Comments and docstrings translated to English for review; logic unchanged.

For execute_dax and olap_schema it checks that:
  - without the olap permission the tool is NOT REGISTERED in the schema handed to
    the model at all — not "present, but 403";
  - a runtime call without the permission is refused (the second half of the
    double gate);
  - after a web_search in the same turn both tools are blocked.

And for the prompt block:
  - it is injected only when the olap permission is present.

Assertions match Russian substrings of production messages, so those stay as they
are; their English meaning is given in an adjacent comment.

Run: python tests/test_olap_permissions.py
"""

from __future__ import annotations

import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db  # noqa: E402
import prompts  # noqa: E402

GROUP = "permissions-prompt-unit"
results: list[tuple[str, str, str]] = []
TOOLS = ("execute_dax", "olap_schema")


def rec(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, "PASS" if ok else "FAIL", detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def _names(permissions: dict | None) -> set[str]:
    return {tool["function"]["name"] for tool in db.tools_for(permissions)}


def test_not_registered_without_permission() -> None:
    without = _names({})
    without_none = _names(None)
    with_olap = _names({"olap": True})
    for tool in TOOLS:
        ok = tool not in without and tool not in without_none and tool in with_olap
        rec(f"{tool}: not registered without the permission, present with it", ok,
            f"no permissions={tool in without}, None={tool in without_none}, "
            f"with olap={tool in with_olap}")


def test_runtime_gate() -> None:
    dax = db.tool_execute_dax("EVALUATE {1}", permissions={})
    schema = db.tool_olap_schema(permissions={})
    dax_none = db.tool_execute_dax("EVALUATE {1}", permissions=None)
    schema_none = db.tool_olap_schema(permissions=None)
    # "прав" is the stem of "permissions" in the production refusal message.
    ok = all("error" in r and "прав" in r["error"].lower()
             for r in (dax, schema, dax_none, schema_none))
    rec("runtime call without the permission is refused (fail-closed, incl. permissions=None)", ok,
        f"execute_dax={dax.get('error')}, olap_schema={schema.get('error')}")


def test_web_guard() -> None:
    for tool in TOOLS:
        blocked = db.dispatch_tool(tool, {"query": "EVALUATE {1}"},
                                   permissions={"olap": True}, web_search_used=True)
        # "веб-поиск" is "web search" in the production refusal message.
        ok = "error" in blocked and "веб-поиск" in blocked["error"].lower()
        rec(f"{tool}: blocked after a web_search in the same turn", ok,
            blocked.get("error", ""))


def test_web_guarded_set() -> None:
    ok = all(tool in db.WEB_GUARDED_TOOLS for tool in TOOLS)
    rec("both tools are in WEB_GUARDED_TOOLS", ok, str(sorted(db.WEB_GUARDED_TOOLS)))


def test_prompt_gate() -> None:
    base = prompts.build_system_prompt()
    with_olap = prompts.build_system_prompt(olap=True)
    others = prompts.build_system_prompt(websearch=True, rag_read=True, file_read=True)
    ok = (prompts.OLAP_CUBE_OVERVIEW not in base
          and prompts.OLAP_CUBE_OVERVIEW in with_olap
          and prompts.OLAP_CUBE_OVERVIEW not in others)
    rec("the prompt block is injected only with the olap permission", ok,
        f"no permissions={prompts.OLAP_CUBE_OVERVIEW in base}, "
        f"with olap={prompts.OLAP_CUBE_OVERVIEW in with_olap}")


def test_prompt_has_current_date() -> None:
    """The date is computed when the prompt is built: we patch "today" and look for
    it in the text. The expected string is the Russian "Today is 29 July 2026."
    produced by the production prompt builder."""
    original = prompts._today  # noqa: SLF001
    prompts._today = lambda: date(2026, 7, 29)  # noqa: SLF001
    try:
        fixed = prompts.build_system_prompt()
        with_olap = prompts.build_system_prompt(olap=True)
    finally:
        prompts._today = original  # noqa: SLF001
    live = prompts.build_system_prompt()
    today = date.today()
    ok = ("Сегодня 29 июля 2026 года." in fixed
          and "Сегодня 29 июля 2026 года." in with_olap
          and f"{today.day} {prompts._MONTHS[today.month - 1]} {today.year}" in live)
    rec("the prompt carries today's date, computed at build time", ok,
        f"patched: {'Сегодня 29 июля 2026 года.' in fixed}, "
        f"unpatched: line about {today.isoformat()}")


def test_prompt_relative_periods_rule() -> None:
    block = prompts.OLAP_CUBE_OVERVIEW
    # Russian fragments of rule 12 in the cube prompt block: "Relative periods",
    # "from today's date", "name the concrete period in your answer".
    ok = ("Относительные периоды" in block
          and "от сегодняшней даты" in block
          and "назови в ответе конкретный период" in block)
    rec("the cube block carries the rule about relative periods", ok,
        "rule 12 is in place" if ok else "the rule is missing")


TESTS = [
    test_not_registered_without_permission,
    test_runtime_gate,
    test_web_guard,
    test_web_guarded_set,
    test_prompt_gate,
    test_prompt_has_current_date,
    test_prompt_relative_periods_rule,
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
