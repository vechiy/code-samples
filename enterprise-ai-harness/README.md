# AI assistant: the request harness (excerpt from a production system)

A corporate chat assistant with tools: sales-cube analytics, search over internal
documents, web search, reading attachments, image generation. Around 210 users and
a set of 19 boolean permissions instead of roles. What is shown here is **one path
of one request**, from the permission check to the answer, not the whole system.

Two satellite services share this assistant's SSO and sit behind the same
permission flags: a cosmetics formulation assistant and a legal contract-analysis
RAG (about 5,000 active contracts, structured term extraction, cross-document
comparison). Both run entirely on a local GPU host (Ollama, qwen-family models),
so no masking gateway is involved: the data never leaves the network. They are
separate codebases and are not part of this excerpt.

Every request to a cloud LLM goes through an external masking service: it replaces
names, counterparties and other identifiers with tokens of the form
`[[CUSTOMER_4B319A8C1CF7]]`. The service is external because the substitution
dictionary must not live inside an application that talks to the internet.
**Fail-closed here is meant literally:** if the service is unavailable or returns
`blocked`, an exception is raised before the tool-calling loop is entered, that is,
before the single place where an HTTP request to the provider is made. A masking
failure means the request fails, not that raw text is sent.

The `sessionId` is issued by the masking service on every `tokenize` call. A single
answer spans several such sessions: one for the user message and one more for each
tool result. A token restored through a **foreign** session comes back not as the
original substring but as the canonical directory entry: that is, as a different
legal entity. This is why `SessionTokenMap` remembers which session each token was
born in and restores each one through that session; a token with no session of its
own is left in the text as is. A silent substitution in analytics is worse than a
visible breakage.

Tools return `{"error": "..."}` and never raise out: a tool response is a message to
the model, not an answer to the user. The model has to see the refusal, explain it
to the human and, if it makes sense, try a different way. A failing tool that took
down the SSE stream would cut off the whole answer. The technical cause goes to
stderr; the model gets neutral text with no table or column names in it.

`rag_excerpt.py` is the document access control of the RAG path. The rag_read
permission is checked at the tool entry, rag_write and admin at the HTTP layer, and
the set of collections a user can read (public plus per-user grants) is resolved in
two places: in `rag_collections` for the upload target, and as a separate SQL
statement inside `rag_search` for the retrieval filter. The remaining RAG routes,
the collection CRUD and the grant management are not shown.

`usage_costs.py` is the cost formula, included whole. A stored `cost_rub`, in the
`llm_usage` row or in the JSON result of a tool, is returned as is; computation
happens only where no stored value exists: tokens times an editable price list for
LLM calls, a static per-image price plus a per-reference surcharge for images, and
no value for video. The price list contents and the code that loads them are not
shown.

**Start with `llm_excerpt.py`**: that is the harness itself. Then `db_excerpt.py`
(the double permission gate and the dispatcher), then `crypto.py` (the contract with
the masking gateway), then `rag_excerpt.py` and `usage_costs.py`. The tests in
`tests/` are hermetic: no network, no database.
`test_chat_fail_closed.py` breaks the masking service on purpose and asserts that
the provider is never called and that nothing is streamed to the user.

Comments and docstrings were translated to English for review; the logic is
unchanged. Strings that production sends to the model or shows to the user are
still in Russian. The tests assert on some of them, and each one carries its
English meaning in an adjacent comment.

Known gaps of this path are discussed in the application text, not here.

```
User        -> FastAPI    : POST /chats/{id}/messages (JWT)
FastAPI     -> auth       : resolve user, load permission flags from DB
FastAPI     -> registry   : tools_for(permissions) -> tool schemas (empty = no permission)
FastAPI     -> harness    : chat_with_tools_streaming(messages, schemas, permissions)
harness     -> masking    : tokenize(every message)      [failure -> STOP, nothing leaves]
harness     -> LLM provider: POST (masked conversation)  [+ usage recorded]
LLM provider-> harness    : tool_call(name, arguments)
harness     -> dispatcher : permission gate -> guard -> audit -> tool
harness     -> masking    : tokenize(tool result)        [failure -> STOP]
harness     -> User       : restore(final answer) -> SSE chunks
```
