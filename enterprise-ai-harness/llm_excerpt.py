"""ВЫДЕРЖКА из backend/llm.py рабочей системы. Не самостоятельный модуль.

Что включено (нумерация строк — оригинального файла):
    67-186   class SessionTokenMap — учёт того, какая сессия маскирования
             породила каждый токен, и восстановление каждого токена своей
             сессией.
    472-740  chat_with_tools_streaming — цикл tool-calling для SSE-чата:
             барьеры маскирования, вызов провайдера, учёт расхода,
             диспетчеризация инструментов.

Что опущено (и почему файл не импортируется как есть):
    1-66     импорты, конфигурация провайдеров, MAX_TOOL_ITERATIONS,
             _TOKEN_RE, _BATCH_SEPARATOR;
    187-293  реестр моделей (ModelInfo, списки моделей), классы исключений
             LLMProviderUnavailable / MaskingBlockedError, user_facing_error;
    294-345  _post_to_llm — HTTP к провайдеру с ретраем только транспортных сбоев;
    346-470  chat_with_tools — несходная (не-SSE) версия того же цикла.

Зависимости, которых здесь тоже нет: crypto (включён в сэмпл отдельным файлом),
db (включён выдержкой db_excerpt.py), usage_storage, analyze_uploaded.

Изменения относительно оригинала — только удаление и замена
конфиденциальных строк: наименования контрагентов в докстринге
SessionTokenMap заменены на вымышленные, имя облачного провайдера в
комментарии обезличено. Логика не менялась.
"""



class SessionTokenMap:
    """Кто из сессий породил каждый токен — и восстановление каждого своей.

    Зачем это нужно. В одном ответе сессий несколько: одна на сообщение
    пользователя и ещё по одной на каждый результат инструмента. Раньше
    восстановление перебирало сессии подряд и отдавало сервису весь текст: первая
    сессия натыкалась на токены, рождённые во второй, и подставляла вместо них
    каноническую запись справочника. До своей сессии очередь доходила, когда
    подставлять уже нечего.

    Пока сервис резолвил токен одинаково в любой сессии, это не проявлялось.
    После его исправления от 05.08 своя сессия отдаёт исходную подстроку, а чужая
    каноническую запись, и порядок перебора стал определять результат: на живых
    запросах имена совпадали с исходными в 5, 7 и 9 случаях из 10.
    Воспроизведено: тот же токен через свою сессию даёт «Alpha Trading LLC»,
    через чужую — «Alpha Trading LLC (Грузополучатель, Москва)», то есть
    ДРУГОЕ юрлицо.

    Поэтому здесь запоминается, в какой сессии родился каждый токен, и каждый
    восстанавливается только своей. Перебора всех сессий подряд больше нет.
    """

    def __init__(self, request_id: str) -> None:
        self._request_id = request_id
        # токен → сессия, в которой он впервые появился
        self._owner: dict[str, str] = {}
        self._sessions: list[str] = []

    @property
    def sessions(self) -> list[str]:
        """Порядок появления сессий. Нужен только логам: на восстановление
        он больше не влияет."""
        return list(self._sessions)

    def remember(self, masked_text: str, session_id: str | None) -> None:
        """Связать токены из маскированного текста с их сессией.

        Первая сессия выигрывает: один и тот же токен, встреченный дважды,
        обозначает одно значение, и перепривязка ничего не изменила бы.
        """
        if not session_id:
            return
        if session_id not in self._sessions:
            self._sessions.append(session_id)
        for token in _TOKEN_RE.findall(masked_text or ""):
            self._owner.setdefault(token, session_id)

    def _restore_batch(self, tokens: list[str], session_id: str) -> dict[str, str]:
        """Восстановить токены одной сессии. Возвращает токен → значение.

        Пакетом, а не по одному вызову на токен: ответ с таблицей на десятки
        строк дал бы десятки round-trip'ов к сервису. Если число строк в ответе
        не совпало с числом токенов — разбор пакета ненадёжен, и мы честно
        откатываемся на поштучное восстановление, а не гадаем по позициям.
        """
        payload = _BATCH_SEPARATOR.join(tokens)
        try:
            restored = crypto.restore(
                text=payload, session_id=session_id, request_id=self._request_id
            ).restored_text
        except crypto.CryptoError as exc:
            print(f"[RESTORE] сессия {session_id[:8]}: пакет не восстановлен: {exc}",
                  file=sys.stderr)
            return {}

        parts = restored.split(_BATCH_SEPARATOR)
        if len(parts) == len(tokens):
            return dict(zip(tokens, parts))

        print(
            f"[RESTORE] сессия {session_id[:8]}: пакет вернул {len(parts)} строк "
            f"на {len(tokens)} токенов — поштучно",
            file=sys.stderr,
        )
        values: dict[str, str] = {}
        for token in tokens:
            try:
                values[token] = crypto.restore(
                    text=token, session_id=session_id, request_id=self._request_id
                ).restored_text
            except crypto.CryptoError as exc:
                print(f"[RESTORE] токен {token} не восстановлен: {exc}", file=sys.stderr)
        return values

    def restore(self, text: str) -> str:
        """Восстановить текст: каждый токен — сессией, в которой он родился.

        Токен, для которого своей сессии не нашлось, остаётся в тексте как есть.
        Подставить его чужой сессией нельзя: именно так и появляется чужое
        юрлицо вместо запрошенного, а тихая подмена в аналитике продаж хуже
        видимой поломки — её никто не заметит. Сам факт уходит в stderr.
        """
        if not text:
            return text

        found = _TOKEN_RE.findall(text)
        if not found:
            return text

        by_session: dict[str, list[str]] = {}
        orphans: list[str] = []
        seen: set[str] = set()
        for token in found:
            if token in seen:
                continue
            seen.add(token)
            owner = self._owner.get(token)
            if owner is None:
                orphans.append(token)
                continue
            by_session.setdefault(owner, []).append(token)

        if orphans:
            print(
                f"[RESTORE] без своей сессии, оставлены как есть: {orphans}",
                file=sys.stderr,
            )

        values: dict[str, str] = {}


def chat_with_tools_streaming(
    messages: list[dict[str, Any]],
    model: ModelInfo,
    on_event: Callable[[str, dict[str, Any]], None],
    temperature: float = 0.3,
    attachments: list[crypto.Attachment] | None = None,
    tool_results: list[dict[str, Any]] | None = None,
    usage_user_id: int | None = None,
    tools_schema: list[dict[str, Any]] | None = None,
    author_permissions: dict[str, Any] | None = None,
    image_model_id: str | None = None,
    image_reference_ids: list[str] | None = None,
    video_model_id: str | None = None,
) -> str:
    """
    SSE-friendly tool loop.

    MVP note: tool calls are resolved synchronously, preserving the masking flow
    from chat_with_tools. The final restored answer is emitted as chunk events.

    image_model_id — выбор модели генерации картинок из UI. Едет сквозь цикл
    как непрозрачное значение и проверяется по белому списку уже в туле; LLM
    его не видит и повлиять на него не может.

    image_reference_ids — прикреплённые пользователем картинки-образцы (в порядке
    прикрепления) для
    генерации по референсу (img2img). Едет тем же боковым каналом и по той же
    причине: что приложено, решает человек, а не содержимое чата. Проверку
    владельца и поддержку моделью делает тул.

    video_model_id — выбор модели генерации видео из UI. Тот же боковой канал:
    у моделей видео цена отличается в разы, и выбирать её по тексту переписки
    нельзя. Белый список проверяет тул.
    """
    request_id = f"chat-{uuid.uuid4().hex[:12]}"
    user_id = "fastapi"
    tokens = SessionTokenMap(request_id)
    # Маскированные версии вложений (для тула read_uploaded): storage_ref → маска,
    # file_name → оригинальное имя (модель матчит по нему). Оригинал сюда не идёт.
    masked_attachments: list[crypto.Attachment] = []

    # Документы (PDF/DOCX/TXT) сервис маскирования как вложение НЕ принимает (HTTP 400) —
    # маскированной копии файла для них не бывает. Поэтому в tokenize уходят только
    # таблицы, а документы едут в тул как оригиналы: их текст маскируется барьером
    # результата тула ниже (tokenize_text «результат инструмента», fail-closed).
    # Отправить документ в tokenize нельзя не только из принципа: это уронило бы чат.
    table_attachments = [a for a in (attachments or []) if not crypto.is_document(a)]
    doc_attachments = [a for a in (attachments or []) if crypto.is_document(a)]

    def tokenize_text(
        text: str,
        what: str,
        message_attachments: list[crypto.Attachment] | None = None,
    ) -> str:
        if not text:
            return text

        try:
            res = crypto.tokenize(
                prompt=text,
                user_id=user_id,
                request_id=request_id,
                # Любой облачный backend (любой из облачных провайдеров) = external → данные
                # маскируются. Локально остаётся только ollama. Не ослаблять.
                external_llm=(model.backend != "ollama"),
                llm_model=model.id,
                attachments=message_attachments,
            )
        except crypto.CryptoError as e:
            raise RuntimeError(f"Ошибка маскирования {what}: {e}") from e

        if res.blocked:
            raise MaskingBlockedError(
                f"Сервис маскирования заблокировал {what}: "
                f"{', '.join(res.block_reasons)}"
            )

        # Токены запоминаются вместе со своей сессией — по ним и восстанавливаем.
        tokens.remember(res.masked_prompt, res.session_id)

        # Захват маскированных версий вложений (для read_uploaded): путь берём
        # маскированный (res.masked_attachments), имя — оригинальное (модель
        # матчит по нему). Оригинальный storage_ref вниз не уходит.
        if message_attachments and res.masked_attachments:
            originals = {a.attachment_id: a for a in message_attachments}
            for masked in res.masked_attachments:
                original = originals.get(masked.attachment_id)
                masked_attachments.append(
                    crypto.Attachment(
                        attachment_id=masked.attachment_id,
                        file_name=(original.file_name if original else masked.file_name),
                        storage_ref=masked.storage_ref,
                        content_type=masked.content_type,
                        file_type=masked.file_type,
                    )
                )

        return res.masked_prompt

    def restore_all(text: str) -> str:
        return tokens.restore(text)

    # Гейт тулов на регистрации: список тулов для модели задаёт вызывающий
    # (db.tools_for(permissions)). Без него — базовый db.TOOLS_SCHEMA (без OLAP).
    active_tools = tools_schema if tools_schema is not None else db.TOOLS_SCHEMA

    # HARD GUARD против prompt injection: как только в этом turn был web_search,
    # корп-data-тулы (db.WEB_GUARDED_TOOLS) блокируются до конца turn. Флаг живёт
    # весь вызов и НЕ сбрасывается между проходами цикла.
    web_search_used = False

    convo: list[dict[str, Any]] = []
    last_user_index = max(
        (index for index, message in enumerate(messages) if message.get("role") == "user"),
        default=-1,
    )
    for index, message in enumerate(messages):
        copied = dict(message)
        if isinstance(copied.get("content"), str) and copied["content"]:
            message_attachments = (
                table_attachments if index == last_user_index else None
            )
            copied["content"] = tokenize_text(
                copied["content"],
                "сообщение",
                message_attachments=message_attachments,
            )
        convo.append(copied)

    # Схема табличных вложений -> в контекст (для тула analyze_uploaded): модель видит
    # колонки/типы/примеры МАСКИРОВАННОГО файла, а не весь файл. Строится из уже
    # маскированных вложений (masked_attachments), само маскирование не трогаем.
    if masked_attachments and last_user_index >= 0:
        import analyze_uploaded

        schema_blocks: list[str] = []
        for att in masked_attachments:
            try:
                sch = analyze_uploaded.schema_of(att.storage_ref, att.file_type)
            except Exception:  # noqa: BLE001 — схема не критична, не роняем чат
                continue
            cols = "; ".join(f"{c['name']}: {c['dtype']}" for c in sch["columns"])
            sheets = sch.get("sheets")
            active = sch.get("active_sheet")
            if sheets:
                # Многолистовой xlsx: перечисляем ВСЕ листы (иначе не-первые молча невидимы).
                # Размеры «сырые» (включая строку заголовка). Колонки/примеры — по active-листу.
                sheets_line = "; ".join(
                    f"{s['name']} ({s['rows']}×{s['cols']})" for s in sheets
                )
                sheets_block = (
                    f"листы xlsx (имя (строк×колонок, с заголовком)): {sheets_line}\n"
                    f"df по умолчанию — лист «{active}»; для другого листа вызови "
                    f"analyze_uploaded с параметром sheet.\n"
                )
                cols_label = f"колонки листа «{active}»"
            else:
                sheets_block = ""
                cols_label = "колонки"
            schema_blocks.append(
                f"Схема прикреплённого файла «{att.file_name}» "
                f"(маскированная; для инструмента analyze_uploaded, переменная df):\n"
                f"{sheets_block}"
                f"строк данных: {sch['total_rows']}\n"
                f"{cols_label}: {cols}\n"
                f"примеры (первые строки):\n{sch['sample_text']}"
            )
        if schema_blocks:
            base = convo[last_user_index].get("content") or ""
            convo[last_user_index]["content"] = base + "\n\n" + "\n\n".join(schema_blocks)

    # Документ схемы не имеет — модели нужна лишь подсказка, чем его читать. Имён файлов
    # здесь НЕ повторяем: они уже перечислены в самом сообщении (и там замаскированы),
    # а второй, немаскированный экземпляр имени сбивал бы матчинг вложения.
    if doc_attachments and last_user_index >= 0:
        base = convo[last_user_index].get("content") or ""
        convo[last_user_index]["content"] = base + (
            "\n\n[Среди вложений есть документы (PDF/DOCX/TXT). Их текст читается "
            "инструментом read_uploaded по имени файла — он отдаёт начало текста. "
            "Расчёты (analyze_uploaded) по документам недоступны.]"
        )

    for _ in range(MAX_TOOL_ITERATIONS):
        payload = {
            "model": model.id,
            "messages": convo,
            "temperature": temperature,
            "tools": active_tools,
        }

        data = _post_to_llm(payload, model.backend)
        usage_storage.log_usage(
            feature="chat",
            provider=model.backend,
            model=(data.get("model") or model.id),
            user_id=usage_user_id,
            request_id=request_id,
            usage=data.get("usage"),
        )
        msg = data["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            raw = msg.get("content") or "(пустой ответ модели)"
            display = restore_all(raw)
            for char in display:
                on_event("chunk", {"text": char})
            return display

        convo.append({
            "role": "assistant",
            "content": msg.get("content") or "",
            "tool_calls": tool_calls,
        })

        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            fn = function.get("name") or "unknown"

            try:
                args = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}

            try:
                restored_args = json.loads(restore_all(json.dumps(args, ensure_ascii=False)))
            except json.JSONDecodeError:
                restored_args = args

            on_event("tool_call", {"name": fn, "args": restored_args})
            result = db.dispatch_tool(
                fn,
                restored_args,
                permissions=author_permissions,
                web_search_used=web_search_used,
                user_id=usage_user_id,
                # Таблицы — маскированные копии, документы — оригиналы (маскируются
                # текстом на выходе тула). Ни один немаскированный байт в LLM не идёт.
                attachments=masked_attachments + doc_attachments,
                image_model_id=image_model_id,
                image_reference_ids=image_reference_ids,
                video_model_id=video_model_id,
            )
            # Ставим флаг ПОСЛЕ диспатча web_search: сам web_search проходит, а
            # любой корп-тул дальше по этому turn (этот батч ниже или след. проход)
            # уже блокируется hard-guard'ом в dispatch_tool.
            if fn == "web_search":
                web_search_used = True
            if tool_results is not None:
                tool_results.append({
                    "tool_name": fn,
                    "args": restored_args,
                    "result": result,
                })
            on_event("tool_result", {"name": fn, "result": result})

            tool_text_raw = json.dumps(result, ensure_ascii=False, default=str)
            tool_text_masked = tokenize_text(tool_text_raw, "результат инструмента")

            convo.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": tool_text_masked,
            })

    final_text = "Превышен лимит итераций инструментов."
    on_event("chunk", {"text": final_text})
    return final_text
