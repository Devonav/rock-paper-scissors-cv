"""Rock-Paper-Scissors core logic: gesture classification and game rules.

This package holds the camera-free, testable heart of the app. The webcam loop
and UI live in the top-level ``main.py``; everything here is pure Python so it
can be unit-tested without a camera or MediaPipe.
"""

from rps.game import AI, Score, Match, decide_winner, CHOICES
from rps.gesture import classify

__all__ = [
    "AI",
    "Score",
    "Match",
    "decide_winner",
    "CHOICES",
    "classify",
]
