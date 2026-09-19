# Code samples

Two excerpts from production systems at a mid-size company, prepared for an external technical review. Architecture, task decomposition, test design and acceptance are mine; the code was written by claude working from my specifications. Comments and docstrings are translated to english for review.
Logic is unchanged, and strings that production sends to the model or shows to the user stay in russian with an adjacent gloss.

## What is here

**`enterprise-ai-harness/`**: one request path through a corporate chat
assistant with tools (around 200 users). Permission gate at registration and at
runtime, fail-closed masking before any cloud llm call, session-aware token
restore, a single tool dispatcher, hermetic negative tests. Start with
`enterprise-ai-harness/README.md`.

**`manager-talk-analyzer/`**: the evaluation side of a call-analysis pipeline
(ASR, diarization, llm extraction into a fixed schema, score computed by code
from model-filled criteria). A golden reference with per-segment label
provenance, a contradiction judge with a synthetic-corruption control and its
measured false-positive background, a blind A/B of two ASR engines, and a
prompt-injection experiment on the extraction step with text and recorded-audio
conditions. Start with `manager-talk-analyzer/README.md`, then
`manager-talk-analyzer/mats-test/EXPERIMENT.md`.

## How to read

The two folders are two sides of one profile: the first shows how to put control
boundaries around llm in production, the second shows how to measure what such
a component actually does. Each README names what is deliberately left out of
the excerpt and what was unfinished. Full repositories are
available on request.

Nothing here runs standalone: production dependencies (databases, masking
service, the GPU inference host) are not included. The tests in
`enterprise-ai-harness/tests/` are hermetic and readable as specifications.

Planned next step for both folders: a minimal runnable subset with synthetic fixtures and stubs, so the security invariants and the experiment gate can be executed without production dependencies.

Also: public mcp server for 1C ERP analytics (sprosierp.ru) is a separate project, not included here
