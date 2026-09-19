#!/usr/bin/env python3
"""Excerpt: the score. Its constants, the function that computes it, and where it lands.

Comments and docstrings translated to English for review; logic unchanged.

WHERE IT COMES FROM
    src/mta/pipeline.py, lines 43-50   -> the formula constants
    src/mta/pipeline.py, lines 225-243 -> _force_service_fields()
    src/mta/pipeline.py, lines 246-270 -> compute_score()
    src/mta/files.py,    line 34       -> INTERNAL_CONTACT_TYPE

WHAT WAS OMITTED
    - the rest of pipeline.py: ASR, diarization, role labelling, the LLM call with its
      retry loop and schema validation, database writes, the mail path, the CLI;
    - the Identity dataclass (pipeline.py lines 71-84): the identification result, an
      employee id, the method that produced it (voice, filename, folder, email, none)
      and, for voice, the cosine similarity to the reference. Only `identity.as_dict()`
      is called here;
    - the caller. _force_service_fields() runs on every model answer inside the retry
      loop, before the answer is validated against schemas/sales.json.

WHAT CHANGED RELATIVE TO THE ORIGINAL
    - INTERNAL_CONTACT_TYPE is defined here instead of being imported from
      src/mta/files.py, which is not part of this excerpt;
    - the `identity: Identity` annotation is kept verbatim and stays valid because of
      `from __future__ import annotations`: annotations are never evaluated here;
    - nothing else. Apart from comments and docstrings the two functions and the
      constants are character for character the original, which an AST comparison
      against pipeline.py confirms.

WHY THESE TRAVEL TOGETHER
    compute_score is the arithmetic; _force_service_fields is its only call site and the
    reason the arithmetic cannot be argued with. The model returns score_breakdown and
    nothing more: code then overwrites `score`, `contact_type`, `identity`, `source_ref`,
    `source_kind` and `employee_id` in the model's own answer, so a score that disagrees
    with its own breakdown has no way to exist. This is also why the injection experiment
    aims at the five breakdown entries: the sum itself is out of the model's reach.

    Two ways out of a number exist, and both are deliberate. contact_type
    "внутренний разговор" is set by code from the telephony numbers, and an internal
    conversation is not scored at all. All four criteria null means the criteria do not
    apply to this communication (a no-answer, a wrong number), and the result is
    score = null rather than a low score. A missing criterion is not a failed one.
"""

from __future__ import annotations

from typing import Any

# src/mta/files.py line 34, kept in Russian: it is a stored contact_type value and an
# enum member in schemas/sales.json, not a message to a person.
INTERNAL_CONTACT_TYPE = "внутренний разговор"

# The score: a base of 2, plus 2 for each criterion met, plus initiative, clamped to the
# range. Computed by code, so that a divergence from the breakdown is impossible.
SCORE_CRITERIA = ("greeting_ok", "needs_discovered", "objections_handled", "next_step_secured")
SCORE_BASE = 2
SCORE_PER_CRITERION = 2
SCORE_MIN = 1
SCORE_MAX = 10
INITIATIVE_MAX = 2


def _force_service_fields(
    result: dict[str, Any],
    *,
    source_kind: str,
    identity: Identity,
    file_name: str,
    timecodes: bool,
    contact_type: str,
) -> dict[str, Any]:
    """The service fields are known to us exactly, so we do not rely on the model."""
    if not isinstance(result, dict):
        return result
    result["source_kind"] = source_kind
    result["employee_id"] = identity.employee
    result["identity"] = identity.as_dict()
    result["source_ref"] = {"file": file_name, "timecodes": timecodes}
    result["contact_type"] = contact_type
    result["score"] = compute_score(result.get("score_breakdown"), contact_type)
    return result


def compute_score(breakdown: Any, contact_type: str) -> int | None:
    """The final score out of score_breakdown. None means there is nothing to score.

    The model supplies the criteria only, the addition is done by code: that makes a
    score diverging from its own breakdown impossible.
    """
    if contact_type == INTERNAL_CONTACT_TYPE:
        return None
    if not isinstance(breakdown, dict):
        return None

    values = [breakdown.get(name) for name in SCORE_CRITERIA]
    if all(value is None for value in values):
        return None

    raw = breakdown.get("initiative")
    initiative = raw if isinstance(raw, int) and not isinstance(raw, bool) else 0
    initiative = max(0, min(INITIATIVE_MAX, initiative))

    total = (
        SCORE_BASE
        + SCORE_PER_CRITERION * sum(1 for value in values if value is True)
        + initiative
    )
    return max(SCORE_MIN, min(SCORE_MAX, total))
