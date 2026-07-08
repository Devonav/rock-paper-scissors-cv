"""Classify a MediaPipe hand into rock / paper / scissors.

MediaPipe returns 21 landmarks per hand. We decide which fingers are
"extended" by comparing fingertip positions against lower joints, then map
the count/pattern of extended fingers to a gesture. This is a rule-based
classifier: no training, no dataset, works out of the box.

Landmark index reference (https://developers.google.com/mediapipe):
    0  wrist
    4  thumb tip          3  thumb IP
    8  index tip          6  index PIP
    12 middle tip        10 middle PIP
    16 ring tip          14 ring PIP
    20 pinky tip         18 pinky PIP
"""

import math

# Fingertip landmark ids and the joint (PIP) two steps below each tip.
# The thumb is handled separately because it bends sideways, not down.
_FINGERS = {
    "index": (8, 6),
    "middle": (12, 10),
    "ring": (16, 14),
    "pinky": (20, 18),
}


def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def _fingers_up(landmarks):
    """Return a dict {finger_name: bool} of which fingers are extended.

    For the four fingers we compare the tip to its PIP joint in image space.
    Because we work on the selfie-flipped frame the hand is upright, so an
    extended finger has its tip *above* (smaller y) the joint. We scale the
    threshold by hand size so it holds whether the hand is near or far.
    """
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    hand_size = _dist(wrist, middle_mcp) or 1e-6  # guard divide-by-zero

    up = {}
    for name, (tip_id, pip_id) in _FINGERS.items():
        tip, pip = landmarks[tip_id], landmarks[pip_id]
        # Positive margin => tip clearly higher than the joint => extended.
        margin = (pip.y - tip.y) / hand_size
        up[name] = margin > 0.15

    # Thumb: extended when the tip is far (horizontally) from the index MCP.
    thumb_tip, index_mcp = landmarks[4], landmarks[5]
    up["thumb"] = (_dist(thumb_tip, index_mcp) / hand_size) > 0.55
    return up


def classify(landmarks):
    """Map a list of 21 landmarks to 'rock', 'paper', 'scissors', or None.

    Returns None when the pose is ambiguous (e.g. one finger up), so callers
    can ask the user to try again instead of guessing.
    """
    up = _fingers_up(landmarks)
    extended = sum((up["index"], up["middle"], up["ring"], up["pinky"]))

    if extended == 0:
        return "rock"  # closed fist (thumb position doesn't matter)
    # Paper: all four fingers out. Allow the thumb to be tucked, and tolerate
    # one finger reading as curled so a slightly relaxed open hand still counts.
    if extended >= 4 or (extended == 3 and up["index"] and up["pinky"]):
        return "paper"
    # Scissors: index + middle up with ring + pinky down. The thumb is ignored
    # (people hold it either way), which is what makes this robust in practice.
    if up["index"] and up["middle"] and not up["ring"] and not up["pinky"]:
        return "scissors"
    return None  # anything else is ambiguous
