"""EXCERPT from backend/llm.py of the production system. Not a standalone module.

Comments and docstrings translated to English for review; logic unchanged.

What is included (line numbers are those of the original file):
    67-186   class SessionTokenMap — tracking which masking session produced each
             token, and restoring every token through its own session.
    472-740  chat_with_tools_streaming — the tool-calling loop behind the SSE chat:
             masking barriers, provider call, usage accounting, tool dispatch.

What is omitted (and why the file does not import as is):
    1-66     imports, provider configuration, MAX_TOOL_ITERATIONS,
             _TOKEN_RE, _BATCH_SEPARATOR;
    187-293  the model registry (ModelInfo, model lists), the exception classes
             LLMProviderUnavailable / MaskingBlockedError, user_facing_error;
    294-345  _post_to_llm — HTTP to the provider, retrying transport failures only;
    346-470  chat_with_tools — the non-streaming version of the same loop.

Dependencies that are also absent here: crypto (shipped as its own file in this
sample), db (shipped as the db_excerpt.py excerpt), usage_storage, analyze_uploaded.

The only changes relative to the original are the removal and substitution of
confidential strings: the counterparty names in the SessionTokenMap docstring were
replaced with fictional ones, and the cloud provider name in a comment was made
generic. The logic is untouched.

Strings that are sent to the model or shown to the user stay in Russian, as in
production; their English meaning is given in an adjacent comment.
"""



class SessionTokenMap:
    """Which session produced each token — and restoring each token through its own.

    Why this is needed. A single answer spans several sessions: one for the user
    message and one more for every tool result. Restoration used to walk the
    sessions in order, handing the whole text to the service each time: the first
    session ran into tokens born in the second and substituted the canonical
    directory entry for them. By the time the owning session got its turn, there
    was nothing left to substitute.

    As long as the service resolved a token identically in any session this stayed
    invisible. After the service was fixed on 05.08, the owning session returns the
    original substring while a foreign one returns the canonical entry, so the walk
    order began to decide the result: on live requests the names matched the
    originals in 5, 7 and 9 cases out of 10. Reproduced: the same token through its
    own session yields "Alpha Trading LLC", through a foreign one
    "Alpha Trading LLC (consignee, Moscow)" — that is, a DIFFERENT legal entity.

    Hence this class remembers which session each token was born in and restores
    each one only through that session. The walk over all sessions is gone.
    """

    def __init__(self, request_id: str) -> None:
        self._request_id = request_id
        # token → the session in which it first appeared
        self._owner: dict[str, str] = {}
        self._sessions: list[str] = []

    @property
    def sessions(self) -> list[str]:
        """The order in which sessions appeared. Only logs need it: it no longer
        affects restoration."""
        return list(self._sessions)

    def remember(self, masked_text: str, session_id: str | None) -> None:
        """Bind the tokens of a masked text to their session.

        First session wins: the same token seen twice denotes the same value, so
        rebinding it would change nothing.
        """
        if not session_id:
            return
        if session_id not in self._sessions:
            self._sessions.append(session_id)
        for token in _TOKEN_RE.findall(masked_text or ""):
            self._owner.setdefault(token, session_id)

    def _restore_batch(self, tokens: list[str], session_id: str) -> dict[str, str]:
        """Restore the tokens of one session. Returns token → value.

        In a batch rather than one call per token: an answer holding a table of
        dozens of rows would mean dozens of round trips to the service. If the
        number of lines in the response does not match the number of tokens, the
        batch cannot be parsed reliably, so we fall back honestly to restoring them
        one by one instead of guessing by position.
        """
        payload = _BATCH_SEPARATOR.join(tokens)
        try:
            restored = crypto.restore(
                text=payload, session_id=session_id, request_id=self._request_id
            ).restored_text
        except crypto.CryptoError as exc:
            print(f"[RESTORE] session {session_id[:8]}: batch not restored: {exc}",
                  file=sys.stderr)
            return {}

        parts = restored.split(_BATCH_SEPARATOR)
        if len(parts) == len(tokens):
            return dict(zip(tokens, parts))

        print(
            f"[RESTORE] session {session_id[:8]}: batch returned {len(parts)} lines "
            f"for {len(tokens)} tokens — falling back to one by one",
            file=sys.stderr,
        )
        values: dict[str, str] = {}
        for token in tokens:
            try:
                values[token] = crypto.restore(
                    text=token, session_id=session_id, request_id=self._request_id
                ).restored_text
            except crypto.CryptoError as exc:
                print(f"[RESTORE] token {token} not restored: {exc}", file=sys.stderr)
        return values

    def restore(self, text: str) -> str:
        """Restore a text: every token through the session it was born in.

        A token with no session of its own is left in the text as is. Substituting
        it through a foreign session is not an option: that is exactly how a
        different legal entity appears in place of the requested one, and a silent
        substitution in sales analytics is worse than a visible breakage — nobody
        would notice it. The fact itself goes to stderr.
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
                f"[RESTORE] no owning session, left as is: {orphans}",
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

    image_model_id — the image generation model picked in the UI. It travels through
    the loop as an opaque value and is checked against an allowlist inside the tool;
    the LLM neither sees it nor can influence it.

    image_reference_ids — reference images attached by the user (in attachment
    order) for reference-based generation (img2img). Same side channel, same
    reason: what is attached is decided by a human, not by the chat content.
    Ownership and model support are checked by the tool.

    video_model_id — the video generation model picked in the UI. Same side channel:
    video model prices differ several-fold, so the model must not be chosen from the
    text of the conversation. The allowlist is checked by the tool.
    """
    request_id = f"chat-{uuid.uuid4().hex[:12]}"
    user_id = "fastapi"
    tokens = SessionTokenMap(request_id)
    # Masked versions of the attachments (for the read_uploaded tool): storage_ref →
    # the masked copy, file_name → the original name (the model matches on it).
    # The original never gets here.
    masked_attachments: list[crypto.Attachment] = []

    # Documents (PDF/DOCX/TXT) are NOT accepted as an attachment by the masking
    # service (HTTP 400) — no masked copy of such a file exists. So only tables go
    # to tokenize, while documents reach the tool as originals: their text is masked
    # by the tool-result barrier below (tokenize_text "tool result", fail-closed).
    # Sending a document to tokenize is not merely wrong in principle: it would
    # bring the chat down.
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
                # Any cloud backend (any of the cloud providers) = external → the data
                # is masked. Only ollama stays local. Do not weaken this.
                external_llm=(model.backend != "ollama"),
                llm_model=model.id,
                attachments=message_attachments,
            )
        except crypto.CryptoError as e:
            raise RuntimeError(f"Masking failed for {what}: {e}") from e

        if res.blocked:
            raise MaskingBlockedError(
                f"The masking service blocked {what}: "
                f"{', '.join(res.block_reasons)}"
            )

        # Tokens are remembered together with their session — that is what we restore by.
        tokens.remember(res.masked_prompt, res.session_id)

        # Capture the masked versions of the attachments (for read_uploaded): the
        # path is the masked one (res.masked_attachments), the name is the original
        # one (the model matches on it). The original storage_ref never goes down.
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

    # Registration-time tool gate: the caller decides the tool list handed to the
    # model (db.tools_for(permissions)). Without it — the base db.TOOLS_SCHEMA
    # (no OLAP).
    active_tools = tools_schema if tools_schema is not None else db.TOOLS_SCHEMA

    # HARD GUARD against prompt injection: as soon as a web_search has happened in
    # this turn, the corporate data tools (db.WEB_GUARDED_TOOLS) are blocked until
    # the end of the turn. The flag lives for the whole call and is NOT reset
    # between loop iterations.
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
                "message",
                message_attachments=message_attachments,
            )
        convo.append(copied)

    # The schema of tabular attachments goes into the context (for the
    # analyze_uploaded tool): the model sees the columns/types/samples of the MASKED
    # file, not the whole file. It is built from the already masked attachments
    # (masked_attachments); masking itself is left alone.
    if masked_attachments and last_user_index >= 0:
        import analyze_uploaded

        schema_blocks: list[str] = []
        for att in masked_attachments:
            try:
                sch = analyze_uploaded.schema_of(att.storage_ref, att.file_type)
            except Exception:  # noqa: BLE001 — the schema is optional, never break the chat
                continue
            cols = "; ".join(f"{c['name']}: {c['dtype']}" for c in sch["columns"])
            sheets = sch.get("sheets")
            active = sch.get("active_sheet")
            if sheets:
                # Multi-sheet xlsx: list ALL sheets (otherwise the non-first ones are
                # silently invisible). Sizes are "raw" (header row included).
                # Columns/samples are those of the active sheet.
                sheets_line = "; ".join(
                    f"{s['name']} ({s['rows']}×{s['cols']})" for s in sheets
                )
                # Prompt text below, sent to the model in Russian, as in production:
                # "xlsx sheets (name (rows×cols, header included))", "df defaults to
                # sheet <active>; for another sheet call analyze_uploaded with the
                # sheet parameter", "columns of sheet <active>", "columns".
                sheets_block = (
                    f"листы xlsx (имя (строк×колонок, с заголовком)): {sheets_line}\n"
                    f"df по умолчанию — лист «{active}»; для другого листа вызови "
                    f"analyze_uploaded с параметром sheet.\n"
                )
                cols_label = f"колонки листа «{active}»"
            else:
                sheets_block = ""
                cols_label = "колонки"
            # Prompt text: "Schema of the attached file <name> (masked; for the
            # analyze_uploaded tool, variable df)", "data rows", "samples (first rows)".
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

    # A document has no schema — the model only needs a hint about what to read it
    # with. File names are NOT repeated here: they are already listed in the message
    # itself (and masked there), and a second, unmasked copy of the name would throw
    # off attachment matching.
    # Prompt text below: "[Some of the attachments are documents (PDF/DOCX/TXT).
    # Their text is read by the read_uploaded tool by file name — it returns the
    # beginning of the text. Computations (analyze_uploaded) are not available for
    # documents.]"
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
            # Fallback shown to the user: "(empty model response)".
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
                # Tables are masked copies, documents are originals (masked as text
                # on the way out of the tool). Not one unmasked byte reaches the LLM.
                attachments=masked_attachments + doc_attachments,
                image_model_id=image_model_id,
                image_reference_ids=image_reference_ids,
                video_model_id=video_model_id,
            )
            # The flag is set AFTER dispatching web_search: web_search itself goes
            # through, while any corporate tool later in this turn (further down this
            # batch or on the next iteration) is already blocked by the hard guard
            # inside dispatch_tool.
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
            tool_text_masked = tokenize_text(tool_text_raw, "tool result")

            convo.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": tool_text_masked,
            })

    # Shown to the user: "The tool iteration limit has been exceeded."
    final_text = "Превышен лимит итераций инструментов."
    on_event("chunk", {"text": final_text})
    return final_text
