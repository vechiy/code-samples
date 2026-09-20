# AI assistant: the request harness (excerpt from a production system)

A corporate chat assistant with tools: sales-cube analytics, search over internal
documents, web search, reading attachments, image generation. Around 200 users, and
a set of 19 boolean permissions instead of roles. What is shown here is one path of
one request, from the permission check to the answer, not the whole system.

## Backends

Chat runs against one of three backends, selected per request by the `backend`
field of the model the user picked in the UI. Two are cloud, OpenAI-compatible:
the primary one and a fallback aggregator, chosen at import time by the
`LLM_CLOUD_PROVIDER` switch, which decides which model registry is active. The
third is a local Ollama host with a qwen-family model; it enters the registry only
when `OLLAMA_ENABLED` is set, because it needs the GPU server. The `backend` value
selects the URL and the headers in `_post_to_llm`; everything before that point in
the loop is the same code for all three.

Image generation goes to a separate Image API of the primary cloud vendor, and
video generation to a further provider that bills in credits. Both are called from
their own modules, not from the chat loop, and both are priced by `usage_costs.py`.

The local route is not exempt from the masking barrier. `tokenize` is called for
every message on every backend; what changes for Ollama is one field in the
request context, `externalLlm`, which goes out as `false` (`llm.py:373` and
`llm.py:536`, `crypto.py:127` and `crypto.py:156`). What the masking service does
with that flag is decided inside that service, which is not part of this excerpt.
In this code there is no branch that skips the barrier.

## Satellites

Two satellite services share this assistant's SSO and sit behind the same
permission flags: a cosmetics formulation assistant and a legal contract-analysis
RAG (about 5,000 active contracts, structured term extraction, cross-document
comparison). Both run entirely on a local GPU host (Ollama, qwen-family models),
so no masking gateway is involved: the data never leaves the network. They are
separate codebases and are not part of this excerpt.

## What this excerpt shows

**One path of one request.** A message arrives over HTTP with a JWT, the user's
permission flags are loaded, the tool schemas are built from those flags, and the
tool-calling loop streams the answer back over SSE. The excerpts here cover that
path and nothing else.

**Masking, fail-closed.** Every request to a cloud LLM goes through an external
masking service that replaces names, counterparties and other identifiers with
tokens of the form `[[CUSTOMER_4B319A8C1CF7]]`. The service is external because the
substitution dictionary must not live inside an application that talks to the
internet. If it is unavailable or returns `blocked`, an exception is raised before
the tool-calling loop is entered, that is, before the single place where an HTTP
request to the provider is made. A masking failure means the request fails, not
that raw text is sent.

**`sessionId` and `SessionTokenMap`.** The masking service issues a `sessionId` on
every `tokenize` call, and one answer spans several of them: one for the user
message and one more for each tool result. A token restored through a foreign
session comes back not as the original substring but as the canonical directory
entry: a different legal entity. `SessionTokenMap` records which session each token
was born in and restores each one through that session; a token with no session of
its own is left in the text as is. A silent substitution in analytics is worse than
a visible breakage.

**The double permission gate.** A tool the user has no permission for is not in the
schema sent to the model at all, and the tool function checks the same permission
again when called. Both halves fail closed: absent or `None` permissions mean
refusal. The registration half is in `tools_for`, the runtime half in each `tool_*`
function.

**The dispatcher, and errors as values.** One function routes every tool call. It
carries a hard guard: once a web search has happened in a turn, the corporate data
tools are refused for the rest of that turn. Tools return `{"error": "..."}` and
never raise out, because a tool response is a message to the model, not an answer
to the user. The model has to see the refusal and explain it. The technical cause
goes to stderr; the model gets text with no table or column names in it.

**Document access in RAG.** The `rag_read` permission is checked at the tool entry,
`rag_write` and `admin` at the HTTP layer, and the set of collections a user can
read (public plus per-user grants) is resolved in two places: in `rag_collections`
for the upload target, and as a separate SQL statement inside `rag_search` for the
retrieval filter. The two queries are independent and would drift apart silently;
the file header says so.

**Usage accounting.** A stored `cost_rub`, in the `llm_usage` row or in the JSON
result of a tool, is returned as is. Computation happens only where no stored value
exists: tokens times an editable price list for LLM calls, a static per-image price
plus a per-reference surcharge for images, and no value for video. Four readers
call the same function, so the numbers cannot diverge between screens.

## Files and reading order

1. `llm_excerpt.py`: the harness itself, the masking barriers and the tool loop.
2. `db_excerpt.py`: the double permission gate and the tool dispatcher.
3. `crypto.py`: the contract with the masking gateway, whole module.
4. `rag_excerpt.py`: document access control, assembled from four files.
5. `usage_costs.py`: the cost formula, whole module.
6. `tests/test_chat_fail_closed.py`: breaks the masking service on purpose and
   asserts that the provider is never called and nothing is streamed to the user.
7. `tests/test_olap_permissions.py`: both halves of the gate for the two OLAP
   tools, and the web-search guard.
8. `tests/test_image_gen_permissions.py`: the gate for image generation, masking
   as a stop-validator, and the path check on serving a generated file.

The tests are hermetic: no network, no database. They are readable as
specifications of the invariants above.

## What is deliberately not here

The frontend, the database migrations, the route layer beyond the two RAG gates,
and the remaining tools (OLAP transport, web search, attachment reading, the pandas
sandbox, image and video generation). Each excerpt header lists the line ranges it
was taken from and what was dropped from them.

## Translation note

Comments and docstrings were translated to English for review; the logic is
unchanged. Strings that production sends to the model or shows to the user are
still in Russian. The tests assert on some of them, and each one carries its
English meaning in an adjacent comment.

## Known gaps

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
