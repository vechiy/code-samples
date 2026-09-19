# MTA: code sample

MTA analyses sales phone calls: ASR, diarization, and an LLM step that extracts a
JSON document from the transcript. The manager's score is not the model's: the model
fills in criteria, and code sums them by a fixed formula.

What is shown here is the evaluation side. A golden reference where every label
carries its provenance, a changelog of corrections and one turn marked unresolvable
with the measurement behind it. A contradiction judge: a separate model call that
compares a transcript with a finished extraction, with a control that injects
synthetic corruption. A blind A/B of two ASR engines, including the residual leak
that could not be hidden. And the metrics these are read with: WER with an
anonymisation wildcard, time-weighted speaker attribution, turn structure.

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

Reading order: `tests/golden/golden_call-01.json`, then `scripts/january_selfcheck.py`,
then `build_blind` in `scripts/asr_ab_excerpt.py`, then `mats-test/RESULTS.md`.
