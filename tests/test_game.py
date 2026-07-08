"""Tests for the pure game logic: rules, AI, scoring, and match play.

These need no camera or MediaPipe — they exercise `game.py` directly, which is
why that module is kept free of any OpenCV/MediaPipe imports.
"""

import pytest

from rps import AI, Score, Match, decide_winner, CHOICES
from rps.game import _BEATS, _COUNTER


# --------------------------------------------------------------------------
# decide_winner
# --------------------------------------------------------------------------
@pytest.mark.parametrize("player, ai, expected", [
    ("rock", "scissors", "player"),
    ("scissors", "paper", "player"),
    ("paper", "rock", "player"),
    ("scissors", "rock", "ai"),
    ("paper", "scissors", "ai"),
    ("rock", "paper", "ai"),
    ("rock", "rock", "tie"),
    ("paper", "paper", "tie"),
    ("scissors", "scissors", "tie"),
])
def test_decide_winner(player, ai, expected):
    assert decide_winner(player, ai) == expected


def test_beats_and_counter_are_consistent():
    # Every move beats exactly one and is countered by exactly one.
    assert set(_BEATS) == set(CHOICES)
    for winner, beaten in _BEATS.items():
        assert _COUNTER[beaten] == winner
        # Playing the counter of X against X should win.
        assert decide_winner(_COUNTER[beaten], beaten) == "player"


# --------------------------------------------------------------------------
# AI
# --------------------------------------------------------------------------
def test_random_ai_only_returns_valid_moves():
    ai = AI(adaptive=False, seed=0)
    moves = {ai.move() for _ in range(200)}
    assert moves <= set(CHOICES)
    # With 200 draws we should have seen all three at least once.
    assert moves == set(CHOICES)


def test_random_ai_never_predicts():
    ai = AI(adaptive=False, seed=0)
    for _ in range(50):
        ai.move()
        assert ai.last_prediction is None


def test_adaptive_ai_learns_a_transition():
    """If the player always plays rock after paper, the AI should learn to
    predict rock in that state and throw its counter (paper)."""
    ai = AI(adaptive=True, seed=1)
    # Train on a strict paper->rock->...->paper->rock pattern.
    for mv in ["paper", "rock"] * 10:
        ai.move()
        ai.observe(mv)
    # Force the "just played paper" state and check the prediction.
    ai._history[-1] = "paper"
    assert ai._predict() == "rock"


def test_adaptive_ai_needs_evidence_before_predicting():
    ai = AI(adaptive=True, seed=1)
    assert ai._predict() is None          # no history at all
    ai.observe("rock")
    assert ai._predict() is None          # one move, no transition seen yet


def test_adaptive_ai_bluffs_sometimes():
    """Over many rounds against a perfectly predictable player, the AI should
    still occasionally NOT counter — that's the bluff keeping it unpredictable."""
    ai = AI(adaptive=True, seed=7)
    countered = bluffed = 0
    for _ in range(300):
        ai.observe("rock")               # player is 100% rock
        mv = ai.move()
        if ai.last_prediction is None:
            bluffed += 1
        elif mv == _COUNTER["rock"]:
            countered += 1
    assert countered > 0
    assert bluffed > 0                    # bluff rate ~25%, so this must trigger


def test_ai_reset_clears_memory():
    ai = AI(adaptive=True, seed=1)
    for mv in ["rock", "paper", "scissors"]:
        ai.observe(mv)
    ai.reset()
    assert ai._history == []
    assert ai.last_prediction is None
    assert ai._predict() is None


def test_ai_ignores_invalid_observations():
    ai = AI(adaptive=True, seed=1)
    ai.observe("banana")
    ai.observe(None)
    assert ai._history == []


# --------------------------------------------------------------------------
# Score
# --------------------------------------------------------------------------
def test_score_counts_and_rounds():
    s = Score()
    for r in ["player", "player", "ai", "tie"]:
        s.update(r)
    assert (s.player, s.ai, s.ties) == (2, 1, 1)
    assert s.rounds == 4


def test_streak_builds_and_flips():
    s = Score()
    s.update("player")
    s.update("player")
    assert s.streak == 2
    s.update("ai")                        # flips to AI side
    assert s.streak == -1
    s.update("ai")
    assert s.streak == -2
    s.update("player")                    # flips back
    assert s.streak == 1


def test_tie_preserves_streak():
    s = Score()
    s.update("player")
    s.update("tie")
    assert s.streak == 1                   # a tie does not reset the streak


# --------------------------------------------------------------------------
# Match
# --------------------------------------------------------------------------
def test_best_of_must_be_positive_odd():
    for bad in (0, -1, 2, 4):
        with pytest.raises(ValueError):
            Match(best_of=bad)


@pytest.mark.parametrize("best_of, needed", [(1, 1), (3, 2), (5, 3), (7, 4)])
def test_wins_needed(best_of, needed):
    assert Match(best_of).wins_needed == needed


def test_match_decides_at_threshold():
    m = Match(best_of=3)                   # first to 2
    assert m.update("player") is None      # 1-0, still going
    assert not m.over
    assert m.update("player") == "player"  # 2-0, decided
    assert m.over
    assert m.winner == "player"


def test_ties_do_not_advance_match():
    m = Match(best_of=3)
    for _ in range(5):
        assert m.update("tie") is None
    assert m.player_wins == m.ai_wins == 0
    assert not m.over


def test_match_ignores_updates_after_over():
    m = Match(best_of=1)                   # first to 1
    assert m.update("ai") == "ai"
    # Further updates must not change a settled match.
    m.update("player")
    m.update("player")
    assert m.winner == "ai"
    assert m.player_wins == 0


def test_match_reset():
    m = Match(best_of=3)
    m.update("player")
    m.update("ai")
    m.reset()
    assert m.player_wins == 0 and m.ai_wins == 0
    assert not m.over
