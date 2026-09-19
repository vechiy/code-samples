#!/usr/bin/env python3
"""Unit tests for the stitching rules: micro-alignment, anti-noise inserts, timbre arbiter.

Comments and docstrings translated to English for review; logic unchanged.
Russian fixture words ("раз", "да", "нет" and so on) are test data, not messages,
and are left as they are.

There is no test framework in this project, so this is a standalone script built on
bare asserts. Run it as: python tests/test_merge_rules.py (it is part of
scripts/smoke.sh). Everything is synthetic: no audio, no models, no database.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mta import merge  # noqa: E402
from mta import asr  # noqa: E402
from mta.asr import Word, trim_words  # noqa: E402
from mta.diarize import SpeakerSegment  # noqa: E402

FRAME = 0.02
PASSED: list[str] = []


def check(name: str) -> None:
    PASSED.append(name)
    print(f"  ok  {name}")


def mask_from_runs(total_s: float, runs: list[tuple[float, float]]) -> list[bool]:
    frames = int(total_s / FRAME)
    mask = [False] * frames
    for start, end in runs:
        for i in range(int(start / FRAME), min(frames, int(end / FRAME))):
            mask[i] = True
    return mask


def test_micro_align_moves_word_to_own_segment() -> None:
    """A word standing on another speaker's segment moves onto a speech chunk of its own."""
    mask = mask_from_runs(3.0, [(0.0, 1.0), (1.30, 1.50)])
    segments = [SpeakerSegment(0.0, 1.10, "A"), SpeakerSegment(1.28, 1.52, "B")]
    assigned = [(Word(0.0, 1.0, "раз"), "A"), (Word(0.95, 1.20, "нет"), "B")]
    result, moves = merge.micro_align(
        assigned, segments, mask, frame_s=FRAME, window_s=0.5, only={1}
    )
    assert len(moves) == 1, moves
    moved = result[1][0]
    assert 1.28 <= moved.start < 1.52, moved
    assert moved.text == "нет"
    check("micro-alignment: the word moved onto its own speaker's segment")

    # Same word, but there is no segment of its own nearby: do not move it.
    far = [SpeakerSegment(0.0, 1.10, "A"), SpeakerSegment(2.60, 2.90, "B")]
    result2, moves2 = merge.micro_align(
        assigned, far, mask, frame_s=FRAME, window_s=0.5, only={1}
    )
    assert moves2 == [], moves2
    assert result2[1][0].start == 0.95
    check("micro-alignment: with no segment inside the window the word stays put")

    # The word already stands on its own segment: nothing to do.
    ok_assigned = [(Word(0.0, 1.0, "раз"), "A"), (Word(1.30, 1.48, "нет"), "B")]
    _, moves3 = merge.micro_align(
        ok_assigned, segments, mask, frame_s=FRAME, window_s=0.5, only={1}
    )
    assert moves3 == [], moves3
    check("micro-alignment: a word already on its own segment is left alone")


def test_unclear_insert_and_antinoise() -> None:
    """A placeholder appears on speech and does not appear on noise."""
    mask = mask_from_runs(4.0, [(0.0, 1.0), (2.00, 2.40), (3.0, 4.0)])
    words = [Word(0.0, 1.0, "раз"), Word(3.0, 4.0, "два")]
    utts = [merge.Utterance("A", 0.0, 1.0, "раз"), merge.Utterance("A", 3.0, 4.0, "два")]
    segments = [SpeakerSegment(0.0, 1.0, "A"), SpeakerSegment(1.95, 2.45, "B"),
                SpeakerSegment(3.0, 4.0, "A")]

    result, added = merge.insert_unclear(
        utts, segments, words, mask, frame_s=FRAME, min_speech_s=0.15
    )
    assert len(added) == 1 and added[0]["speaker"] == "B", added
    assert any(u.text == merge.UNCLEAR_TEXT and u.pinned for u in result)
    check("insert: 0.4 s of speech without words became a placeholder utterance")

    _, added_strict = merge.insert_unclear(
        utts, segments, words, mask, frame_s=FRAME, min_speech_s=0.5
    )
    assert added_strict == [], added_strict
    check("anti-noise: below the energy threshold no placeholder is inserted")

    quiet = mask_from_runs(4.0, [(0.0, 1.0), (3.0, 4.0)])
    _, added_quiet = merge.insert_unclear(
        utts, segments, words, quiet, frame_s=FRAME, min_speech_s=0.15
    )
    assert added_quiet == [], added_quiet
    check("anti-noise: a segment without speech energy yields no placeholder")

    same_speaker = [SpeakerSegment(0.0, 1.0, "A"), SpeakerSegment(1.95, 2.45, "A"),
                    SpeakerSegment(3.0, 4.0, "A")]
    _, added_same = merge.insert_unclear(
        utts, same_speaker, words, mask, frame_s=FRAME, min_speech_s=0.15
    )
    assert added_same == [], added_same
    check("anti-noise: a gap inside the same speaker's speech is not a dialogue turn")

    no_mask, added_none = merge.insert_unclear(
        utts, segments, words, None, frame_s=FRAME, min_speech_s=0.15
    )
    assert added_none == [] and no_mask == utts
    check("without a speech mask there are no inserts at all")


def test_arbiter_margin_and_candidates() -> None:
    """The arbiter keeps quiet without a margin and does not reach inside a long segment."""
    assigned = [(Word(0.0, 0.3, "а"), "A"), (Word(0.3, 0.6, "б"), "A"),
                (Word(0.6, 0.9, "в"), "A")]
    # word 0: the other profile is closer by only 0.06, that is noise, not a reason to
    # repaint; word 2: its own profile wins by a wide margin, a confident confirmation.
    scores = {0: {"A": 0.50, "B": 0.56}, 2: {"A": 0.80, "B": 0.50}}
    result, pinned, moved, moves = merge.apply_timbre(assigned, scores, margin=0.10)
    assert moves == [], moves                      # margin 0.06 < 0.10, leave it alone
    assert result[0][1] == "A"
    assert 2 in pinned                             # this one, though, is a confident confirmation
    check("arbiter: a margin below the threshold produces no reattribution")

    result2, pinned2, moved2, moves2 = merge.apply_timbre(assigned, scores, margin=0.05)
    assert len(moves2) == 1 and result2[0][1] == "B", moves2
    assert 0 in pinned2 and moved2 == {0}, (pinned2, moved2)
    assert moved == set(), moved
    check("arbiter: a margin above the threshold moves the word and pins it")

    segments = [SpeakerSegment(0.0, 2.0, "A"), SpeakerSegment(2.0, 2.2, "B")]
    deep = [(Word(0.5, 0.8, "внутри"), "A"), (Word(2.0, 2.2, "край"), "B")]
    picked = merge.arbiter_candidates(deep, segments)
    assert 0 not in picked, picked
    assert 1 in picked, picked
    check("arbiter: a word deep inside its own long segment is not checked")


def test_pinned_survives_smoothing() -> None:
    """A micro-utterance confirmed by timbre does not dissolve into its neighbour."""
    assigned = [(Word(0.0, 1.0, "раз"), "A"), (Word(1.0, 1.1, "да"), "B"),
                (Word(1.2, 2.0, "два"), "A")]
    pinned_utts = merge.build_utterances(assigned, {1})
    kept = merge.smooth_utterances(pinned_utts, min_utter_s=0.3, merge_gap_s=1.0)
    assert [u.speaker for u in kept] == ["A", "B", "A"], kept
    check("smoothing: the pinned micro-utterance survived")

    plain = merge.smooth_utterances(
        merge.build_utterances(assigned), min_utter_s=0.3, merge_gap_s=1.0
    )
    assert [u.speaker for u in plain] == ["A"], plain
    check("smoothing: an unpinned micro-utterance still dissolves")


def test_trim_words() -> None:
    """Trimming removes the tail that falls on silence and does not squeeze a word to a point."""
    mask = mask_from_runs(2.0, [(0.0, 0.30), (1.00, 1.02)])
    words = [Word(0.0, 0.9, "раз"), Word(1.0, 1.9, "два")]
    trimmed = trim_words(words, mask, FRAME)
    assert abs(trimmed[0].end - 0.30) < 1e-6, trimmed[0]
    assert trimmed[1].end - trimmed[1].start >= 0.10 - 1e-9, trimmed[1]
    assert trimmed[1].start == 1.00
    check("trimming: the tail on silence is cut, the minimum length is respected")

    silent = trim_words([Word(0.5, 0.7, "шум")], mask_from_runs(2.0, []), FRAME)
    assert silent[0].start == 0.5 and silent[0].end == 0.7
    check("trimming: a word with no speech inside it is left untouched")


def test_signed_roles() -> None:
    """Roles are signed by code when the employee's side and the call direction are known."""
    speakers = ["SPEAKER_00", "SPEAKER_01"]
    assert merge.signed_roles("SPEAKER_00", speakers, "internal") == {
        "SPEAKER_00": "manager", "SPEAKER_01": "colleague"}
    assert merge.signed_roles("SPEAKER_00", speakers, "queue") == {
        "SPEAKER_00": "manager", "SPEAKER_01": "colleague"}
    check("roles: an internal call yields a colleague")

    assert merge.signed_roles("SPEAKER_01", speakers, "in") == {
        "SPEAKER_01": "manager", "SPEAKER_00": "client"}
    assert merge.signed_roles("SPEAKER_01", speakers, "out") == {
        "SPEAKER_01": "manager", "SPEAKER_00": "client"}
    check("roles: an external call yields a client")

    assert merge.signed_roles("SPEAKER_00", speakers, "unknown") is None
    assert merge.signed_roles("SPEAKER_00", speakers, None) is None
    assert merge.signed_roles(None, speakers, "in") is None
    assert merge.signed_roles("SPEAKER_09", speakers, "in") is None
    assert merge.signed_roles("SPEAKER_00", speakers + ["SPEAKER_02"], "in") is None
    check("roles: without a direction, without a side, and for three speakers we do not sign, the LLM decides")

    labels = merge.role_label_map(merge.signed_roles("SPEAKER_00", speakers, "in"), speakers)
    assert labels == {"SPEAKER_00": "МЕНЕДЖЕР", "SPEAKER_01": "КЛИЕНТ"}, labels
    check("roles: transcript labels match what the mechanics module expects")


def test_mechanics_reads_stored_utterances() -> None:
    """Mechanics takes the stored utterances and does not reassemble them from raw material."""
    from mta import mechanics, pipeline
    from mta.config import load_config

    cfg = load_config()
    utts = [merge.Utterance("SPEAKER_00", 0.0, 4.0, "здравствуйте, чем помочь?"),
            merge.Utterance("SPEAKER_01", 5.0, 9.0, "нужен счёт"),
            merge.Utterance("SPEAKER_00", 10.0, 14.0, "сделаем сегодня")]
    labels = merge.role_label_map(
        merge.signed_roles("SPEAKER_00", ["SPEAKER_00", "SPEAKER_01"], "in"),
        ["SPEAKER_00", "SPEAKER_01"],
    )
    row = {
        "utterances": pipeline.utterances_payload(utts),
        "transcript": merge.render_transcript(utts, labels),
        # raw material is deliberately absent: the old path would have crashed on this
        "words": [], "speaker_segments": [], "duration_s": 20.0,
    }
    roles, restored = mechanics.speaker_roles(cfg, row)
    assert roles == {"SPEAKER_00": "МЕНЕДЖЕР", "SPEAKER_01": "КЛИЕНТ"}, roles
    assert len(restored) == 3
    check("mechanics: stored utterances are read, no raw material required")

    payload = mechanics.compute(cfg, dict(row, speaker_segments=[
        {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"},
        {"start": 5.0, "end": 9.0, "speaker": "SPEAKER_01"},
        {"start": 10.0, "end": 14.0, "speaker": "SPEAKER_00"},
    ]))
    assert payload.get("version"), payload
    check("mechanics: metrics are computed on the stored utterances")

    legacy = {"utterances": None, "transcript": "", "words": [], "speaker_segments": []}
    try:
        mechanics.speaker_roles(cfg, legacy)
    except mechanics.MechanicsError:
        check("mechanics: an old document without raw material still gives a clear error")
    else:
        raise AssertionError("expected MechanicsError for a document without raw material")


def test_overlap_words_limiter() -> None:
    """An overlap zone is declared for short words only."""
    segments = [SpeakerSegment(0.0, 3.0, "A"), SpeakerSegment(1.0, 4.0, "B")]
    short = [(Word(1.2, 1.6, "да"), "A")]
    assert merge.overlap_words(short, segments) == {0}
    check("overlap: a short word in the overlap zone gets a reduced margin")

    long_word = [(Word(1.2, 2.6, "документы"), "A")]
    assert merge.overlap_words(long_word, segments) == set()
    check("overlap: a word longer than 1.0 s is not taken by the rule")

    lone = [(Word(0.2, 0.6, "алло"), "A")]
    assert merge.overlap_words(lone, segments) == set()
    check("overlap: outside the overlap zone the rule does not fire")


def test_level_skew_gate() -> None:
    """The normalisation gate fires only on a skew above the threshold."""
    gold = asr.level_skew_db({"S1": -30.8, "S2": -16.5})
    assert gold == 14.3, gold
    assert gold > asr.NORMALIZE_SKEW_DB
    check("gate: the golden call skew of 14.3 dB is above the threshold, the input is normalised")

    control = asr.level_skew_db({"S1": -20.1, "S2": -26.6})
    assert control == 6.5, control
    assert control <= asr.NORMALIZE_SKEW_DB
    check("gate: the control call skew of 6.5 dB is below the threshold, the input stays raw")

    assert asr.level_skew_db({"S1": -20.0}) is None
    assert asr.level_skew_db({}) is None
    check("gate: without two speakers the skew is not computed and there is no normalisation")


def main() -> int:
    for test in (test_micro_align_moves_word_to_own_segment,
                 test_unclear_insert_and_antinoise,
                 test_arbiter_margin_and_candidates,
                 test_pinned_survives_smoothing,
                 test_trim_words,
                 test_signed_roles,
                 test_mechanics_reads_stored_utterances,
                 test_overlap_words_limiter,
                 test_level_skew_gate):
        print(test.__doc__.splitlines()[0])
        test()
    print(f"\nchecks in total: {len(PASSED)}, all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
