"""Tests for the rule-based gesture classifier.

We don't need a camera or a real hand: `classify` only cares about the
geometry of 21 landmark points, so we synthesise landmark sets that produce a
chosen "fingers up" pattern and assert the resulting gesture. The helper mirrors
the reference layout in gesture.py (wrist=0, mcp joints, pip joints, tips).
"""

from types import SimpleNamespace

import pytest

from rps.gesture import classify, _fingers_up

# (tip id, pip id) for the four non-thumb fingers, matching gesture._FINGERS.
_FINGER_IDS = {
    "index": (8, 6),
    "middle": (12, 10),
    "ring": (16, 14),
    "pinky": (20, 18),
}


def _pt(x, y):
    return SimpleNamespace(x=x, y=y)


def make_hand(fingers_up, thumb_up=False):
    """Build 21 landmarks that yield the requested finger pattern.

    Coordinates are in MediaPipe's normalised image space (y grows downward),
    and the hand is upright (as it is on the selfie-flipped frame), so an
    extended finger has its tip *above* (smaller y) its pip joint.
    """
    lm = [_pt(0.5, 0.5) for _ in range(21)]
    lm[0] = _pt(0.50, 0.90)    # wrist
    lm[9] = _pt(0.50, 0.50)    # middle MCP -> hand_size = 0.40, comfortably big
    lm[5] = _pt(0.45, 0.60)    # index MCP (thumb distance reference)

    for name, (tip, pip) in _FINGER_IDS.items():
        lm[pip] = _pt(0.50, 0.60)
        # Extended: tip well above pip. Curled: tip below pip.
        lm[tip] = _pt(0.50, 0.30) if name in fingers_up else _pt(0.50, 0.64)

    # Thumb: "up" means tip far horizontally from the index MCP.
    lm[4] = _pt(0.90, 0.60) if thumb_up else _pt(0.47, 0.60)
    return lm


# --------------------------------------------------------------------------
# The synthetic-hand helper itself must produce the intended pattern.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("pattern", [
    set(),
    {"index"},
    {"index", "middle"},
    {"index", "middle", "ring", "pinky"},
])
def test_make_hand_matches_requested_fingers(pattern):
    up = _fingers_up(make_hand(pattern))
    for finger in _FINGER_IDS:
        assert up[finger] is (finger in pattern), finger


def test_thumb_detection():
    assert _fingers_up(make_hand(set(), thumb_up=True))["thumb"] is True
    assert _fingers_up(make_hand(set(), thumb_up=False))["thumb"] is False


# --------------------------------------------------------------------------
# classify
# --------------------------------------------------------------------------
def test_rock_is_closed_fist():
    assert classify(make_hand(set())) == "rock"
    # Thumb position must not matter for rock.
    assert classify(make_hand(set(), thumb_up=True)) == "rock"


def test_paper_is_open_hand():
    assert classify(make_hand({"index", "middle", "ring", "pinky"})) == "paper"


def test_paper_tolerates_one_relaxed_finger():
    # Three fingers with index+pinky out still reads as paper (relaxed open hand).
    assert classify(make_hand({"index", "ring", "pinky"})) == "paper"


def test_scissors_is_index_and_middle():
    assert classify(make_hand({"index", "middle"})) == "scissors"
    # Thumb is ignored for scissors, so either way works.
    assert classify(make_hand({"index", "middle"}, thumb_up=True)) == "scissors"


@pytest.mark.parametrize("ambiguous", [
    {"index"},                     # single finger
    {"pinky"},                     # single finger
    {"index", "ring"},             # split, not a clean V
    {"middle", "ring", "pinky"},   # three without index -> not paper's shape
])
def test_ambiguous_returns_none(ambiguous):
    assert classify(make_hand(ambiguous)) is None
