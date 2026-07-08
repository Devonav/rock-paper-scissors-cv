# Rock Paper Scissors — AI Computer Vision



<video src="https://raw.githubusercontent.com/Devonav/rock-paper-scissors-cv/main/gameplay_web.mp4" width="100%" controls autoplay loop muted></video>



Play Rock Paper Scissors against an AI using your webcam. A countdown ticks
**3 → 2 → 1 → SHOOT!**; during a short capture window your gesture is sampled
across several frames and **majority-voted**, so one blurry frame can't spoil
the round. The AI reveals its move and the winner is scored automatically, with
a live scoreboard, win rate, and win/loss streak.

Hand detection uses [MediaPipe](https://ai.google.dev/edge/mediapipe)'s
HandLandmarker (21 hand landmarks); the gesture is decided by a lightweight
rule-based classifier — no dataset or training required.

## Setup

```bash
pip3 install -r requirements.txt
```

Download the hand model once (~7.8 MB):

```bash
curl -sSL -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## Play

```bash
python3 main.py
```

To record your gameplay to a video file (e.g., for sharing or demonstrations):

```bash
python3 main.py --record
# Or specify a custom filename: python3 main.py --record my_demo.mp4
```

| Key       | Action                                             |
| --------- | -------------------------------------------------- |
| `SPACE`   | Start a round (or start a new match once one ends) |
| `A`       | Toggle adaptive AI (learns your habits)            |
| `M`       | Cycle match format — best of 1 / 3 / 5 / 7         |
| `R`       | Reset the score and match                          |
| `Q`/`ESC` | Quit                                               |

Hold your hand up so the camera can see it, press **SPACE**, and make your
move before the countdown ends. The live `reading:` label at the bottom
shows what the camera currently reads (green once it's confident), so you can
check your pose is being recognized before you commit.

Play is organized into **matches**: the pips at the top of the screen fill in
as each side wins rounds, and the first to take the majority wins the match.
Press **M** between matches to change the format.

## Gestures

| Gesture   | Hand pose                        |
| --------- | -------------------------------- |
| Rock      | Closed fist                      |
| Paper     | Open hand, all fingers extended  |
| Scissors  | Index + middle fingers up (a "V")|

## How it works

| Path                | Responsibility                                             |
| ------------------- | ---------------------------------------------------------- |
| `main.py`           | Webcam loop, round state machine, on-screen UI             |
| `rps/gesture.py`    | 21 landmarks → `rock` / `paper` / `scissors` (or `None`)   |
| `rps/game.py`       | RPS rules, AI opponent, scorekeeping, best-of-N match play |
| `camera_check.py`   | Headless webcam + detection self-test (run if unsure)      |
| `tests/`            | `pytest` suite for the game logic and gesture classifier   |

The `rps/` package is the camera-free core (`from rps import AI, classify, …`);
`main.py` and `camera_check.py` are the entry points that wire it to the webcam.

The classifier decides which fingers are extended by comparing each fingertip
to a lower knuckle, normalized by hand size so it works near or far from the
camera. A fist → rock, an open hand → paper, exactly index+middle up →
scissors; anything ambiguous returns `None` and the round asks you to retry.

Everything in `game.py` and `gesture.py` is pure and camera-free, so it's
covered by fast unit tests:

```bash
pip3 install pytest
pytest
```

## Notes

- Requires a working webcam and a display (uses an OpenCV window).
- `opencv-python` is pinned below 5.0.0 — the 5.0.0 pre-release pairs badly
  with current MediaPipe wheels.
- The AI is fair (uniform random) by default. Press `A` to make it adaptive:
  it models your play as a first-order Markov chain ("after you throw rock, what
  do you usually throw next?") and counters the prediction, bluffing a quarter
  of the time so it never becomes fully predictable. When it guesses, the reveal
  screen tells you what it expected — press `A` again to go back to fair random.
- You can press `SPACE` during the result screen to immediately start the next
  round without waiting for it to clear.
