"""EXCERPT from backend/db.py of the production system. Not a standalone module.

Comments and docstrings translated to English for review; logic unchanged.

What is included (line numbers are those of the original file):
    345-359  WEB_GUARDED_TOOLS — the set of corporate tools that are closed off
             once a web search has happened, plus a note on the symmetric case
             that is still open;
    360-391  tools_for — the REGISTRATION gate: without the permission a tool
             never reaches the schema sent to the model;
    393-495  tool_execute_dax, tool_olap_schema, log_web_search, tool_web_search,
             tool_rag_search — the runtime half of the double gate;
    963-1045 dispatch_tool — the single entry point for calling a tool: hard
             guard against injection, routing, error neutralisation.

The first block is cut at line 495 (end of tool_rag_search) rather than at 500
as originally requested: line 500 falls in the middle of the next signature.

What is omitted:
    1-88     PostgreSQL connection, serialisation, log_query;
    90-344   the JSON schemas of the eight tools that are handed to the model;
    496-962  the remaining tool implementations (reading attachments, the pandas
             sandbox, image and video generation).

Dependencies that are absent here: olap, olap_guards, olap_schema, websearch,
rag_search, read_uploaded, analyze_uploaded, image_client, video.

Nothing was changed relative to the original: no confidential strings were found
in these ranges. The logic is untouched.

User-facing strings returned to the model stay in Russian, as in production —
the tests assert on them. Their English meaning is given in an adjacent comment.
"""


WEB_GUARDED_TOOLS = {
    EXECUTE_DAX_TOOL["function"]["name"],
    OLAP_SCHEMA_TOOL["function"]["name"],
    RAG_SEARCH_TOOL["function"]["name"],
}

# GROUNDWORK (NOT implemented yet): the two-way guard. After rag_search within a
# turn, web_search/execute_dax should be blocked as well — so that internal
# document content cannot leak out (exfiltration through an outbound request).
# That needs a new per-turn rag_used flag threaded through dispatch_tool and the
# tool loop in llm.py (the counterpart of web_search_used) — and llm.py is out of
# scope for this task. Partial protection already exists: web_search has a
# search_guard that inspects the outgoing query for internal/personal data.
# The symmetric guard is a separate task.


def tools_for(permissions: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The tool list handed to the model, scoped to the permissions of the request
    author. The base TOOLS_SCHEMA goes to everyone; execute_dax only when
    olap == true; web_search only when WEBSEARCH_ENABLED AND websearch == true;
    rag_search only when RAG_ENABLED AND rag_read == true; read_uploaded only when
    file_read == true; generate_image only when image_gen == true; generate_video
    only when video == true AND the VIDEO_ENABLED switch is on. None/missing
    permissions → fail-closed."""
    granted = permissions or {}
    tools = list(TOOLS_SCHEMA)
    if granted.get("olap"):
        tools.append(EXECUTE_DAX_TOOL)
        tools.append(OLAP_SCHEMA_TOOL)
    if granted.get("websearch"):
        import websearch
        if websearch.is_enabled():
            tools.append(WEB_SEARCH_TOOL)
    if granted.get("rag_read"):
        import rag_search
        if rag_search.is_enabled():
            tools.append(RAG_SEARCH_TOOL)
    if granted.get("file_read"):
        tools.append(READ_UPLOADED_TOOL)
        tools.append(ANALYZE_UPLOADED_TOOL)
    if granted.get("image_gen"):
        tools.append(GENERATE_IMAGE_TOOL)
    if granted.get("video"):
        from video import factory as video_factory
        if video_factory.is_enabled():
            tools.append(GENERATE_VIDEO_TOOL)
    return tools


def tool_execute_dax(
    query: str,
    permissions: dict[str, Any] | None = None,
    user_id: int | None = None,
) -> dict[str, Any]:
    """OLAP tool: runs DAX against the cube over the configured transport (see olap.py).

    The runtime safety net of the double gate: even if a call somehow reached this
    point, we do NOT touch the cube without an explicit olap == true. None/missing
    permissions → refusal (fail-closed). Before being sent, the query goes through
    the preflight guards (olap_guards): they catch four known model mistakes and
    return text with a ready-made replacement."""
    if not (permissions or {}).get("olap"):
        # "Insufficient permissions for OLAP queries."
        return {"error": "Недостаточно прав для OLAP-запросов."}
    import olap_guards
    blocked = olap_guards.check(query, user_id=user_id)
    if blocked is not None:
        return blocked
    import olap
    return olap.execute_dax(query)


def tool_olap_schema(
    table: str | None = None,
    search: str | None = None,
    full: bool = False,
    permissions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Cube metadata. Same double gate as execute_dax (fail-closed)."""
    if not (permissions or {}).get("olap"):
        # "Insufficient permissions for OLAP queries."
        return {"error": "Недостаточно прав для OLAP-запросов."}
    import olap_schema
    try:
        return olap_schema.describe(table=table, search=search, full=bool(full))
    except Exception as exc:  # noqa: BLE001
        # The model must not see infrastructure detail — it goes to stderr only.
        print(f"[olap_schema] failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        # "The cube structure is temporarily unavailable."
        return {"error": "Структура куба временно недоступна."}


def log_web_search(user_id: int | None, query: str) -> None:
    """Web-search audit trail: who/what/when (results are not stored). Never brings
    the chat down — a write failure only goes to stderr. user_id is mandatory (the
    chat is authenticated); None → skip (a safety net)."""
    if user_id is None:
        print("[websearch] log skipped: user_id is None", file=sys.stderr)
        return
    try:
        _query(
            "INSERT INTO analyst.web_search_log (user_id, query) VALUES (%s, %s)",
            (user_id, query),
        )
    except Exception as exc:  # noqa: BLE001 — the audit must not break search/chat
        print(f"[websearch] audit log not written: {type(exc).__name__}: {exc}",
              file=sys.stderr)


def tool_web_search(
    query: str,
    permissions: dict[str, Any] | None = None,
    user_id: int | None = None,
) -> dict[str, Any]:
    """Web search (see websearch.py). Order: permission gate → audit log →
    search_guard → search. The runtime half of the double gate: without
    websearch == true nothing leaves the perimeter. None/missing permissions →
    refusal (fail-closed)."""
    if not (permissions or {}).get("websearch"):
        # "Insufficient permissions for web search."
        return {"error": "Недостаточно прав для веб-поиска."}

    import websearch
    # The audit entry comes after the permission gate and before search_guard: we
    # record that an authorised user asked at all, even if the guard then blocks
    # the query or the search itself fails.
    log_web_search(user_id, query)

    blocked = websearch.search_guard(query)
    if blocked is not None:
        print(f"[websearch] search_guard blocked the query: {blocked}",
              file=sys.stderr)
        # "The query looks like it contains internal or personal data.
        #  Rephrase it without confidential information."
        return {
            "error": (
                "Запрос похож на содержащий внутренние или персональные данные. "
                "Переформулируйте без конфиденциальных сведений."
            )
        }
    return websearch.web_search(query)


def tool_rag_search(
    query: str,
    permissions: dict[str, Any] | None = None,
    user_id: int | None = None,
) -> dict[str, Any]:
    """RAG retrieval over internal documents (see rag_search.py).

    The runtime half of the double gate: without rag_read == true we do NOT reach
    the documents. None/missing permissions → refusal (fail-closed). A retrieval
    failure (DB/embedder) never brings the chat down — a neutral error goes to the
    model, the real cause goes to stderr."""
    if not (permissions or {}).get("rag_read"):
        # "Insufficient permissions to search the documents."
        return {"error": "Недостаточно прав для поиска по документам."}
    import rag_search
    try:
        return rag_search.rag_search(query, user_permissions=permissions, user_id=user_id)
    except Exception as exc:  # noqa: BLE001 — retrieval must not break the chat
        print(f"[rag] search failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        # "Document search is currently unavailable."
        return {"error": "Поиск по документам сейчас недоступен."}


# (the runtime half of the double gate).
_DISPATCH: dict[str, Any] = {}


def dispatch_tool(
    name: str,
    arguments: dict[str, Any],
    permissions: dict[str, Any] | None = None,
    web_search_used: bool = False,
    user_id: int | None = None,
    attachments: list[Any] | None = None,
    image_model_id: str | None = None,
    image_reference_ids: list[str] | None = None,
    video_model_id: str | None = None,
) -> dict[str, Any]:
    # HARD GUARD: once a web_search has happened in this turn, corporate data tools
    # are forbidden — web content must not reach corporate data within one loop.
    if web_search_used and name in WEB_GUARDED_TOOLS:
        # "After a web search, access to corporate data is not allowed in this
        #  answer. Please make a separate request."
        return {
            "error": (
                "После веб-поиска в этом ответе обращение к корпоративным "
                "данным запрещено. Сделайте отдельный запрос."
            )
        }
    # web_search / execute_dax are deliberately kept out of _DISPATCH: the only way
    # to reach them is through the tool_* wrappers that check permissions (the
    # runtime half of the double gate).
    if name == "web_search":
        return tool_web_search(
            arguments.get("query", ""), permissions=permissions, user_id=user_id
        )
    if name == "execute_dax":
        return tool_execute_dax(
            arguments.get("query", ""), permissions=permissions, user_id=user_id
        )
    if name == "olap_schema":
        return tool_olap_schema(
            arguments.get("table") or None,
            arguments.get("search") or None,
            bool(arguments.get("full")),
            permissions=permissions,
        )
    if name == "rag_search":
        return tool_rag_search(
            arguments.get("query", ""), permissions=permissions, user_id=user_id
        )
    if name == "generate_image":
        return tool_generate_image(
            arguments.get("prompt", ""),
            permissions=permissions,
            user_id=user_id,
            image_model_id=image_model_id,
            image_reference_ids=image_reference_ids,
        )
    if name == "generate_video":
        return tool_generate_video(
            arguments.get("prompt", ""),
            permissions=permissions,
            user_id=user_id,
            video_model_id=video_model_id,
        )
    if name == "read_uploaded":
        return tool_read_uploaded(
            arguments.get("filename", ""),
            permissions=permissions,
            attachments=attachments,
        )
    if name == "analyze_uploaded":
        return tool_analyze_uploaded(
            arguments.get("filename", ""),
            arguments.get("code", ""),
            permissions=permissions,
            attachments=attachments,
            sheet=arguments.get("sheet") or None,
        )
    if name not in _DISPATCH:
        return {"error": f"Unknown tool: {name}"}
    try:
        return _DISPATCH[name](**arguments)
    except Exception as exc:  # noqa: BLE001 — the model gets a clean message
        # The real error goes to the log; the model gets neutral business text with
        # no technical detail (table/column names and the like). The chat does not
        # fall over, it relays this to the user.
        print(f"[tool {name}] {type(exc).__name__}: {exc}", file=sys.stderr)
        return {"error": TOOL_ERROR_MESSAGE}
