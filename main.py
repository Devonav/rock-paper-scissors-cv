"""Rock-Paper-Scissors against an AI, played with your webcam.

Uses MediaPipe's Tasks HandLandmarker API (the current, supported API) to
find 21 hand landmarks per frame; `gesture.classify` turns those into
rock/paper/scissors.

Flow of one round:
    IDLE      -> press SPACE to start
    COUNTDOWN -> "3, 2, 1" shown on screen with a shrinking ring
    SHOOT     -> brief capture window; your gesture is majority-voted here
    REVEAL    -> AI reveals its move; winner shown
    (back to IDLE)

The capture window (rather than a single frame) is what makes detection feel
reliable: we sample your gesture across several frames and take the majority
vote, so one blurry frame can't spoil the round.

Controls:
    SPACE  start a round (or start a new match once one is over)
    A      toggle adaptive AI (learns your habits)
    M      cycle match format (best of 1 / 3 / 5 / 7)
    R      reset the score and match
    Q/ESC  quit
"""

import argparse
import os
import time
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from rps import AI, Match, Score, classify, decide_winner

# Match formats cycled by the M key.
BEST_OF_OPTIONS = (1, 3, 5, 7)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

# --- Tunable timing (seconds) ---
COUNT_STEP = 0.7  # time each of "3","2","1" is shown
SHOOT_HOLD = 0.5  # capture window: gestures sampled and majority-voted
REVEAL_HOLD = 2.8  # how long the result stays before returning to idle

# --- Palette (BGR) ---
WHITE = (245, 245, 245)
DIM = (150, 150, 150)
GREEN = (90, 225, 120)
RED = (70, 80, 240)
YELLOW = (60, 210, 245)
CYAN = (230, 200, 70)
INK = (25, 22, 20)  # panel background

# Emoji-free glyphs for each move, drawn as simple text.
MOVE_LABEL = {"rock": "ROCK", "paper": "PAPER", "scissors": "SCISSORS"}

# Bone connections between the 21 landmarks, for drawing the skeleton.
_HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # index
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # middle
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # ring
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # pinky
    (0, 17),  # palm base
]

FONT = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------
def _panel(frame, x1, y1, x2, y2, alpha=0.55, color=INK):
    """Draw a semi-transparent rounded rectangle in place."""
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    tint = np.full_like(roi, color, dtype=np.uint8)
    frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, tint, alpha, 0)


def _text(frame, text, org, scale, color, thickness=2):
    cv2.putText(frame, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def _text_size(text, scale, thickness):
    (w, h), _ = cv2.getTextSize(text, FONT, scale, thickness)
    return w, h


def _center(frame, text, y, scale, color, thickness=2, shadow=True):
    """Draw horizontally-centered text at baseline y, with a soft shadow."""
    w = frame.shape[1]
    tw, _ = _text_size(text, scale, thickness)
    x = (w - tw) // 2
    if shadow:
        _text(frame, text, (x + 2, y + 2), scale, INK, thickness + 1)
    _text(frame, text, (x, y), scale, color, thickness)


def _draw_hand(frame, landmarks, color=GREEN):
    """Draw the landmark skeleton (Tasks API has no drawing_utils helper)."""
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
    for a, b in _HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (235, 235, 235), 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(frame, (x, y), 5, INK, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)


def _draw_pips(frame, cx, y, filled, total, color):
    """Row of `total` dots centred on cx; the first `filled` are solid."""
    gap = 22
    start = cx - (total - 1) * gap // 2
    for i in range(total):
        x = start + i * gap
        if i < filled:
            cv2.circle(frame, (x, y), 7, color, -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (x, y), 7, DIM, 1, cv2.LINE_AA)


def _draw_scoreboard(frame, score, ai, match, fps):
    """Top HUD: scores, streak, match progress, mode, FPS — one translucent bar."""
    w = frame.shape[1]
    _panel(frame, 0, 0, w, 64, alpha=0.5)

    # Centre: "BEST OF N" with a pip per round needed for each side.
    _center(frame, f"BEST OF {match.best_of}", 22, 0.5, DIM, 1, shadow=False)
    _draw_pips(frame, w // 2 - 90, 46, match.player_wins, match.wins_needed, GREEN)
    _draw_pips(frame, w // 2 + 90, 46, match.ai_wins, match.wins_needed, RED)

    _text(frame, "YOU", (18, 26), 0.6, DIM, 1)
    _text(frame, str(score.player), (18, 54), 0.95, GREEN, 2)

    _text(frame, "AI", (110, 26), 0.6, DIM, 1)
    _text(frame, str(score.ai), (110, 54), 0.95, RED, 2)

    _text(frame, "TIE", (185, 26), 0.6, DIM, 1)
    _text(frame, str(score.ties), (185, 54), 0.95, YELLOW, 2)

    # Win rate over decided rounds.
    decided = score.player + score.ai
    if decided:
        rate = f"{100 * score.player / decided:.0f}% win"
        _text(frame, rate, (270, 26), 0.55, DIM, 1)
    if score.streak:
        s = score.streak
        label = f"W{s}" if s > 0 else f"L{-s}"
        col = GREEN if s > 0 else RED
        _text(frame, f"streak {label}", (270, 52), 0.55, col, 1)

    # Right-aligned mode + FPS.
    mode = "ADAPTIVE" if ai.adaptive else "RANDOM"
    mode_col = CYAN if ai.adaptive else DIM
    mw, _ = _text_size(mode, 0.6, 2)
    _text(frame, mode, (w - mw - 18, 26), 0.6, mode_col, 2)
    fps_txt = f"{fps:2.0f} fps"
    fw, _ = _text_size(fps_txt, 0.5, 1)
    _text(frame, fps_txt, (w - fw - 18, 52), 0.5, DIM, 1)


def _draw_detecting(frame, gesture, confident):
    """Bottom-left pill showing the live-read gesture."""
    label = MOVE_LABEL.get(gesture, "—")
    txt = f"reading: {label}"
    tw, th = _text_size(txt, 0.6, 2)
    x, y = 18, frame.shape[0] - 18
    _panel(frame, x - 8, y - th - 10, x + tw + 10, y + 10, alpha=0.5)
    col = GREEN if confident else DIM
    _text(frame, txt, (x, y), 0.6, col, 2)


def _draw_countdown_ring(frame, fraction, number):
    """Big number in a ring that sweeps as the step elapses.

    `fraction` goes 0 -> 1 across the current step; the ring empties as it does.
    """
    h, w = frame.shape[:2]
    cx, cy, r = w // 2, h // 2, 90
    cv2.circle(frame, (cx, cy), r + 8, INK, -1, cv2.LINE_AA)  # backing disc
    cv2.circle(frame, (cx, cy), r + 8, (60, 60, 60), 3, cv2.LINE_AA)
    end = -90 + int(360 * (1 - fraction))
    cv2.ellipse(frame, (cx, cy), (r, r), 0, -90, end, YELLOW, 6, cv2.LINE_AA)
    tw, tht = _text_size(number, 3.0, 6)
    _text(frame, number, (cx - tw // 2, cy + tht // 2), 3.0, WHITE, 6)


def _draw_reveal(frame, player_move, ai_move, result, prediction):
    """Result screen with both moves side by side and the outcome banner."""
    h, w = frame.shape[:2]
    if player_move is None:
        _panel(frame, w // 2 - 260, h // 2 - 70, w // 2 + 260, h // 2 + 70)
        _center(frame, "No clear hand", h // 2 - 8, 1.2, RED, 3)
        _center(
            frame, "Show rock / paper / scissors and retry", h // 2 + 38, 0.72, WHITE, 2
        )
        return

    banner_txt, banner_col = {
        "player": ("YOU WIN", GREEN),
        "ai": ("AI WINS", RED),
        "tie": ("TIE", YELLOW),
    }[result]

    _panel(frame, w // 2 - 300, h // 2 - 120, w // 2 + 300, h // 2 + 110)
    # "YOU  <move>   vs   AI  <move>"
    line = f"YOU {MOVE_LABEL[player_move]}   vs   AI {MOVE_LABEL[ai_move]}"
    _center(frame, line, h // 2 - 40, 0.95, WHITE, 2)
    _center(frame, banner_txt, h // 2 + 45, 2.1, banner_col, 5)
    if prediction is not None:
        _center(
            frame,
            f"(AI guessed you'd play {MOVE_LABEL[prediction]})",
            h // 2 + 90,
            0.6,
            DIM,
            1,
        )


def _draw_idle_hint(frame):
    h, w = frame.shape[:2]
    txt = "PRESS SPACE TO PLAY"
    tw, th = _text_size(txt, 0.95, 2)
    x = (w - tw) // 2
    _panel(frame, x - 22, h - 66, x + tw + 22, h - 20, alpha=0.5)
    _center(frame, txt, h - 34, 0.95, WHITE, 2)


def _draw_match_over(frame, match):
    """Full-screen match result with a dim overlay and a replay prompt."""
    h, w = frame.shape[:2]
    _panel(frame, 0, 0, w, h, alpha=0.55)  # dim the whole frame
    won = match.winner == "player"
    headline, col = ("YOU WON THE MATCH!", GREEN) if won else ("AI WON THE MATCH", RED)
    _center(frame, headline, h // 2 - 30, 1.7, col, 4)
    _center(
        frame,
        f"{match.player_wins} - {match.ai_wins}  (best of {match.best_of})",
        h // 2 + 25,
        1.0,
        WHITE,
        2,
    )
    _center(frame, "SPACE  new match      M  change format", h // 2 + 80, 0.7, DIM, 2)


# --------------------------------------------------------------------------
# MediaPipe setup
# --------------------------------------------------------------------------
def _make_landmarker():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(
            f"Model file not found: {MODEL_PATH}\n"
            "Download it with:\n"
            "  curl -sSL -o hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"
        )
    base = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = vision.HandLandmarkerOptions(
        base_options=base,
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return vision.HandLandmarker.create_from_options(opts)


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def main(record_path=None):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam (index 0). Is it in use?")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    landmarker = _make_landmarker()
    ai = AI(adaptive=False)
    score = Score()
    best_of_idx = 1  # index into BEST_OF_OPTIONS -> best of 3
    match = Match(BEST_OF_OPTIONS[best_of_idx])

    state = "IDLE"  # IDLE | COUNTDOWN | SHOOT | REVEAL | MATCH_OVER
    phase_start = 0.0  # timestamp the current state began
    player_move = ai_move = result = None
    votes = Counter()  # gestures sampled during the SHOOT window
    last_prediction = None

    # Smooth the live HUD reading so it doesn't flicker frame to frame.
    recent = deque(maxlen=5)
    fps, last_t = 0.0, time.perf_counter()

    window = "Rock Paper Scissors AI"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    writer = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            if record_path and writer is None:
                fh, fw = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(record_path, fourcc, 30.0, (fw, fh))

            frame = cv2.flip(frame, 1)  # selfie view: mirror horizontally
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(time.perf_counter() * 1000)
            result_obj = landmarker.detect_for_video(mp_image, ts_ms)

            raw_gesture = None
            if result_obj.hand_landmarks:
                landmarks = result_obj.hand_landmarks[0]  # list of 21 points
                raw_gesture = classify(landmarks)
                hand_color = CYAN if state == "SHOOT" else GREEN
                _draw_hand(frame, landmarks, hand_color)

            # Majority over the last few frames => stable HUD label.
            recent.append(raw_gesture)
            valid = [g for g in recent if g]
            live_gesture = Counter(valid).most_common(1)[0][0] if valid else None
            confident = bool(valid) and len(valid) >= 3

            now = time.time()

            # ---------------- State machine ----------------
            if state == "COUNTDOWN":
                elapsed = now - phase_start
                # step 0,1,2 -> "3","2","1"; step 3 -> hand off to SHOOT.
                # Guard against COUNT_STEP <= 0 (would divide by zero).
                step = int(elapsed // COUNT_STEP) if COUNT_STEP > 0 else 3
                if step < 3:
                    frac = (elapsed - step * COUNT_STEP) / COUNT_STEP
                    _draw_countdown_ring(frame, min(frac, 1.0), str(3 - step))
                else:
                    state, phase_start = "SHOOT", now
                    votes.clear()

            elif state == "SHOOT":
                _center(frame, "SHOOT!", frame.shape[0] // 2 + 15, 2.4, WHITE, 6)
                if raw_gesture:
                    votes[raw_gesture] += 1
                if now - phase_start >= SHOOT_HOLD:
                    # Majority vote across the capture window.
                    player_move = votes.most_common(1)[0][0] if votes else None
                    ai_move = ai.move()
                    last_prediction = ai.last_prediction
                    if player_move is None:
                        result = "no_hand"
                    else:
                        result = decide_winner(player_move, ai_move)
                        score.update(result)
                        match.update(result)
                        ai.observe(player_move)
                    state, phase_start = "REVEAL", now

            elif state == "REVEAL":
                _draw_reveal(frame, player_move, ai_move, result, last_prediction)
                if now - phase_start > REVEAL_HOLD:
                    # A decided match ends the sequence; otherwise back to idle.
                    state = "MATCH_OVER" if match.over else "IDLE"

            elif state == "MATCH_OVER":
                _draw_match_over(frame, match)

            else:  # IDLE
                _draw_idle_hint(frame)

            # ---------------- FPS (exponential moving average) ----------
            t = time.perf_counter()
            dt = t - last_t
            last_t = t
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            _draw_scoreboard(frame, score, ai, match, fps)
            _draw_detecting(frame, live_gesture, confident)

            if writer:
                # Add a subtle recording indicator
                cv2.circle(frame, (frame.shape[1] - 80, 85), 6, RED, -1, cv2.LINE_AA)
                _text(frame, "REC", (frame.shape[1] - 65, 90), 0.5, RED, 1)
                writer.write(frame)

            cv2.imshow(window, frame)

            # ---------------- Input ----------------
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break
            elif key == ord(" "):
                if state == "MATCH_OVER":
                    match.reset()  # rematch, same format
                    state = "IDLE"
                elif state in ("IDLE", "REVEAL"):
                    state, phase_start = "COUNTDOWN", now
                    player_move = ai_move = result = last_prediction = None
            elif key == ord("a"):
                ai.adaptive = not ai.adaptive
            elif key == ord("m") and state in ("IDLE", "MATCH_OVER"):
                # Change format; only mid-lobby so we never truncate a live match.
                best_of_idx = (best_of_idx + 1) % len(BEST_OF_OPTIONS)
                match = Match(BEST_OF_OPTIONS[best_of_idx])
                score = Score()
                state = "IDLE"
            elif key == ord("r"):
                score = Score()
                match.reset()
                ai.reset()
                state = "IDLE"
    finally:
        if writer:
            writer.release()
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Play Rock Paper Scissors against an AI."
    )
    parser.add_argument(
        "--record",
        nargs="?",
        const="gameplay.mp4",
        help="Record gameplay to the specified video file (default: gameplay.mp4)",
    )
    args = parser.parse_args()
    main(record_path=args.record)
