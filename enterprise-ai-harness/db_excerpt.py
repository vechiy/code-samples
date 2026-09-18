"""ВЫДЕРЖКА из backend/db.py рабочей системы. Не самостоятельный модуль.

Что включено (нумерация строк — оригинального файла):
    345-359  WEB_GUARDED_TOOLS — набор корпоративных инструментов, закрываемых
             после веб-поиска, и комментарий о незакрытом симметричном случае;
    360-391  tools_for — гейт на РЕГИСТРАЦИИ: без права инструмент не попадает
             в схему, отправляемую модели;
    393-495  tool_execute_dax, tool_olap_schema, log_web_search, tool_web_search,
             tool_rag_search — рантайм-половина двойного гейта;
    963-1045 dispatch_tool — единственная точка вызова инструмента: hard-guard
             против инъекции, маршрутизация, нейтрализация ошибок.

Первый блок обрезан на строке 495 (конец tool_rag_search), а не на 500, как в
исходной заявке: 500 приходится на середину сигнатуры следующей функции.

Что опущено:
    1-88     подключение к PostgreSQL, сериализация, log_query;
    90-344   JSON-схемы восьми инструментов, передаваемые модели;
    496-962  реализации остальных инструментов (чтение вложений, песочница
             pandas, генерация изображений и видео).

Зависимости, которых здесь нет: olap, olap_guards, olap_schema, websearch,
rag_search, read_uploaded, analyze_uploaded, image_client, video.

Изменений относительно оригинала нет: конфиденциальных строк в этих диапазонах
не нашлось. Логика не менялась.
"""


WEB_GUARDED_TOOLS = {
    EXECUTE_DAX_TOOL["function"]["name"],
    OLAP_SCHEMA_TOOL["function"]["name"],
    RAG_SEARCH_TOOL["function"]["name"],
}

# ЗАДЕЛ (пока НЕ реализовано): двусторонний guard. После rag_search в этом turn
# стоит блокировать web_search/execute_dax — чтобы внутренний контент документов
# не утёк наружу (эксфильтрация во внешний запрос). Требует нового per-turn флага
# rag_used, прокинутого через dispatch_tool и tool-loop в llm.py (аналог
# web_search_used), — а llm.py в этой задаче не трогаем. Частичная защита уже
# есть: у web_search есть search_guard, инспектирующий исходящий запрос на утечку
# внутренних/персональных данных. Симметричный guard — отдельной задачей.


def tools_for(permissions: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Список тулов для модели под права автора запроса. Базовый TOOLS_SCHEMA —
    всем; execute_dax — только при olap == true; web_search — только при
    WEBSEARCH_ENABLED И websearch == true; rag_search — только при RAG_ENABLED И
    rag_read == true; read_uploaded — только при file_read == true;
    generate_image — только при image_gen == true; generate_video — только при
    video == true И включённом рубильнике VIDEO_ENABLED. None/отсутствие прав →
    fail-closed."""
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
    """OLAP-тул: DAX к кубу выбранным транспортом (см. olap.py).

    Рантайм-страховка двойного гейта: даже если вызов как-то дошёл сюда, без
    явного olap == true к кубу НЕ идём. None/отсутствие прав → отказ (fail-closed).
    Перед отправкой запрос проходит преполётные гарды (olap_guards): они ловят
    четыре известные ошибки модели и возвращают текст с готовой заменой."""
    if not (permissions or {}).get("olap"):
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
    """Метаданные куба. Тот же двойной гейт, что у execute_dax (fail-closed)."""
    if not (permissions or {}).get("olap"):
        return {"error": "Недостаточно прав для OLAP-запросов."}
    import olap_schema
    try:
        return olap_schema.describe(table=table, search=search, full=bool(full))
    except Exception as exc:  # noqa: BLE001
        # Инфраструктуру модель видеть не должна — детали только в stderr.
        print(f"[olap_schema] сбой: {type(exc).__name__}: {exc}", file=sys.stderr)
        return {"error": "Структура куба временно недоступна."}


def log_web_search(user_id: int | None, query: str) -> None:
    """Аудит веб-поиска: кто/что/когда (без результатов). Не роняет чат — сбой
    записи только в stderr. user_id обязателен (чат под авторизацией); None →
    пропуск (страховка)."""
    if user_id is None:
        print("[websearch] log skipped: user_id is None", file=sys.stderr)
        return
    try:
        _query(
            "INSERT INTO analyst.web_search_log (user_id, query) VALUES (%s, %s)",
            (user_id, query),
        )
    except Exception as exc:  # noqa: BLE001 — аудит не должен ронять поиск/чат
        print(f"[websearch] не записал аудит-лог: {type(exc).__name__}: {exc}",
              file=sys.stderr)


def tool_web_search(
    query: str,
    permissions: dict[str, Any] | None = None,
    user_id: int | None = None,
) -> dict[str, Any]:
    """Веб-поиск (см. websearch.py). Порядок: гейт прав → аудит-лог → search_guard
    → поиск. Рантайм-половина двойного гейта: без websearch == true наружу не идём.
    None/отсутствие прав → отказ (fail-closed)."""
    if not (permissions or {}).get("websearch"):
        return {"error": "Недостаточно прав для веб-поиска."}

    import websearch
    # Аудит — после гейта прав, до search_guard: фиксируем сам факт запроса
    # авторизованного юзера, даже если дальше guard заблокирует или поиск упадёт.
    log_web_search(user_id, query)

    blocked = websearch.search_guard(query)
    if blocked is not None:
        print(f"[websearch] search_guard заблокировал запрос: {blocked}",
              file=sys.stderr)
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
    """RAG-ретрив по внутренним документам (см. rag_search.py).

    Рантайм-половина двойного гейта: без rag_read == true к документам НЕ идём.
    None/отсутствие прав → отказ (fail-closed). Сбой поиска (БД/эмбеддер) не
    роняет чат — нейтральный error, реальная причина в stderr."""
    if not (permissions or {}).get("rag_read"):
        return {"error": "Недостаточно прав для поиска по документам."}
    import rag_search
    try:
        return rag_search.rag_search(query, user_permissions=permissions, user_id=user_id)
    except Exception as exc:  # noqa: BLE001 — чат не должен падать из-за ретрива
        print(f"[rag] search failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return {"error": "Поиск по документам сейчас недоступен."}


# (рантайм-половина двойного гейта).
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
    # HARD GUARD: после web_search в этом turn корп-data-тулы запрещены — веб-
    # контент не должен в одном цикле дотянуться до корпоративных данных.
    if web_search_used and name in WEB_GUARDED_TOOLS:
        return {
            "error": (
                "После веб-поиска в этом ответе обращение к корпоративным "
                "данным запрещено. Сделайте отдельный запрос."
            )
        }
    # web_search / execute_dax — вне _DISPATCH намеренно: единственный путь к ним
    # через tool_* с проверкой прав (рантайм-половина двойного гейта).
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
    except Exception as exc:  # noqa: BLE001 — модели отдаём чистое сообщение
        # Реальную ошибку — в лог, модели — нейтральный бизнес-текст без тех-деталей
        # (имён таблиц/колонок и т.п.). Чат не падает, передаёт это пользователю.
        print(f"[tool {name}] {type(exc).__name__}: {exc}", file=sys.stderr)
        return {"error": TOOL_ERROR_MESSAGE}
