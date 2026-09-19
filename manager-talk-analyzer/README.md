# MTA: code sample

MTA analyses sales phone calls: ASR, diarization, and an LLM step that extracts a
JSON document from the transcript. The manager's score is not the model's: the model
fills in criteria, and code sums them by a fixed formula.

The system. In the last three months it processed 7282 call recordings, 254 hours of
8 kHz telephony off a read-only PBX share, and 7303 email threads. It runs inside the
company network: a GPU host serves Ollama (qwen3.5:27b) for extraction and one ASR plus
diarization endpoint, CPU faster-whisper large-v3 is the alternative backend. Nothing is
masked because nothing leaves the network. Roles come from three paths: a voice print
(off on 8 kHz telephony), the internal extension in the telephony file name, and an LLM
call on the transcript. Results land in Postgres as jsonb, read through a local viewer.

What is being evaluated. One LLM call turns a transcript into a JSON document that
has to match `schemas/sales.json`: participants, the client request, each objection and
whether it was handled, agreements, the next step, products, amounts, competitors,
sentiment, confidence, a closed list of missing fields, and five scoring criteria. The
prompt that asks for all of it is `prompts_excerpt.py`, verbatim and untranslated,
because it is what production sends and what the injection cases were aimed at. The
model fills the criteria and never the number: `scoring_excerpt.py` is the code that
sums them, overwrites the service fields in the model's own answer, and decides when
the honest result is no score at all. That split is what makes this step measurable,
because everything the model contributes is a field and every field has a name in the
schema.

What is shown here is the evaluation side. A golden reference where every label
carries its provenance, a changelog of corrections and one turn marked unresolvable
with the measurement behind it. A contradiction judge: a separate model call that
compares a transcript with a finished extraction, with a control that injects
synthetic corruption. A blind A/B of GigaAM-v2 against faster-whisper
large-v3, including the residual leak that could not be hidden: whisper stayed in
production, because GigaAM returns no word level timestamps and speaker attribution is
built on them, although it ran about nine times faster on the same CPU. And the metrics
these are read with: WER with an anonymisation wildcard, time-weighted speaker
attribution, turn structure.

Honest limits. One call is labelled as a golden reference, so this is a pilot of the
method rather than a measured corpus. The judge has never been checked against human
labels: blind forms were prepared for ten calls and have not been filled in. Its v1
control caught 5 of 5 injected corruptions on five cases; v2 in `scripts/selfcheck_v2.py`
caught 15 of 19 applicable corruptions across ten calls, and the same judge returned
20 findings on the untouched parses of those same calls, which is the false positive
floor any "caught N of M" has to be read against.

`mats-test/` is a prompt injection experiment on this extraction step: nine cases,
two conditions (payload pasted into the transcript, payload spoken into the call),
each with a paired filler and a success criterion declared before the run. Results
and what they do and do not show: `mats-test/RESULTS.md`.

Nothing runs as shipped: every entry point needs a .env, a database and recordings
that are not part of this sample. The tests are best read as a specification of the
stitching rules rather than as a suite to execute. The experiment material in
`mats-test/cases/` and `mats-test/bases/` stays in Russian on purpose: its sha256 is
recorded in `mats-test/cases/FROZEN.md` and translating it would break that chain.

## Files

- `schemas/sales.json` the document the extraction call has to produce, verbatim.
- `prompts_excerpt.py` the prompts that ask for it, verbatim and untranslated, each
  block with an English gloss.
- `scoring_excerpt.py` the formula constants, `compute_score`, and the overwrite of the
  service fields in the model's own answer.
- `tests/golden/golden_call-01.json` the golden reference: provenance per segment, a
  changelog of corrections, one turn marked unresolvable with its measurement.
- `tests/test_merge_rules.py` unit tests of the stitching rules (micro-alignment,
  anti-noise inserts, timbre arbiter). Synthetic, no audio and no models.
- `scripts/golden_score.py` the metrics the reference is read with: time-weighted
  speaker attribution, WER with an anonymisation wildcard, speed and silence anomalies.
- `scripts/january_selfcheck.py` the contradiction judge v1: a separate model call that
  reads a transcript against a finished extraction, with a corruption control.
- `scripts/selfcheck_v2.py` the same judge with schema validation, retries, a wider
  corruption set and the false positive denominator.
- `scripts/asr_ab_excerpt.py` what made the A/B blind, and the divergence metric its key
  is read with.
- `mats-test/` the prompt injection experiment: design, frozen cases, runners, results.

## Not included

The pipeline itself (ASR, diarization, stitching, the LLM call with its retries and
schema validation), the web viewer and its API, the database layer and migrations,
voice identification, the audio and the mail, and the raw experiment artifacts
(`results.jsonl` and the model outputs). They are named here so that the gaps are not
read as the whole.

Reading order: this file, then `schemas/sales.json`, then `prompts_excerpt.py`, then
`scoring_excerpt.py`, then `tests/golden/golden_call-01.json`, then
`scripts/january_selfcheck.py`, then `mats-test/EXPERIMENT.md`.
