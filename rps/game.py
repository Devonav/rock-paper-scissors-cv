"""Rock-Paper-Scissors rules and the AI opponent."""

import random

CHOICES = ("rock", "paper", "scissors")

# What each choice beats, and what beats it.
_BEATS = {
    "rock": "scissors",
    "paper": "rock",
    "scissors": "paper",
}
_COUNTER = {beaten: winner for winner, beaten in _BEATS.items()}


def decide_winner(player, ai):
    """Return 'player', 'ai', or 'tie'."""
    if player == ai:
        return "tie"
    return "player" if _BEATS[player] == ai else "ai"


class AI:
    """Picks a move for the computer.

    Default strategy is uniform random (fair and unpredictable). It also keeps
    a memory of the player's moves so it can optionally exploit habits. Set
    `adaptive=True` to enable that.

    The adaptive strategy models the player as a first-order Markov source:
    "given your last move, what do you usually throw next?" That catches the
    real patterns humans fall into (e.g. repeating a winner, switching after a
    loss) far better than just countering your single favorite move. It bluffs
    a fraction of the time so it never becomes fully predictable itself.
    """

    BLUFF_RATE = 0.25   # fraction of adaptive rounds played as pure random

    def __init__(self, adaptive=False, seed=None):
        self.adaptive = adaptive
        self._history = []
        # transitions[a][b] = how often move b followed move a.
        self._transitions = {a: {b: 0 for b in CHOICES} for a in CHOICES}
        self._rng = random.Random(seed)
        # Exposed so the UI can show what the AI is "thinking".
        self.last_prediction = None

    def _predict(self):
        """Best guess at the player's next move, or None if not enough data."""
        if not self._history:
            return None
        counts = self._transitions[self._history[-1]]
        total = sum(counts.values())
        if total < 2:                      # too little evidence for this state
            return None
        # Argmax with deterministic tie-breaking by CHOICES order.
        return max(CHOICES, key=lambda m: counts[m])

    def move(self):
        self.last_prediction = None
        if self.adaptive and self._rng.random() > self.BLUFF_RATE:
            predicted = self._predict()
            if predicted is not None:
                self.last_prediction = predicted
                return _COUNTER[predicted]
        return self._rng.choice(CHOICES)

    def observe(self, player_move):
        """Record the player's move to inform future predictions."""
        if player_move not in CHOICES:
            return
        if self._history:
            self._transitions[self._history[-1]][player_move] += 1
        self._history.append(player_move)

    def reset(self):
        self._history.clear()
        self._transitions = {a: {b: 0 for b in CHOICES} for a in CHOICES}
        self.last_prediction = None


class Score:
    """Running tally across rounds, plus the player's current win streak."""

    def __init__(self):
        self.player = 0
        self.ai = 0
        self.ties = 0
        self.streak = 0        # consecutive player wins (negative = AI streak)

    def update(self, result):
        if result == "player":
            self.player += 1
            self.streak = self.streak + 1 if self.streak >= 0 else 1
        elif result == "ai":
            self.ai += 1
            self.streak = self.streak - 1 if self.streak <= 0 else -1
        else:
            self.ties += 1
            # A tie doesn't break a streak — you kept the AI from scoring.

    @property
    def rounds(self):
        return self.player + self.ai + self.ties


class Match:
    """Best-of-N match: first to `wins_needed` decided rounds takes it.

    Ties don't count toward either side, so a best-of-3 is "first to 2 wins"
    however many ties happen along the way. Reads as done via `winner`.
    """

    def __init__(self, best_of=3):
        if best_of < 1 or best_of % 2 == 0:
            raise ValueError("best_of must be a positive odd number")
        self.best_of = best_of
        self.wins_needed = best_of // 2 + 1
        self.player_wins = 0
        self.ai_wins = 0

    def update(self, result):
        """Record a round result; returns the match winner if just decided."""
        if self.over:
            return self.winner
        if result == "player":
            self.player_wins += 1
        elif result == "ai":
            self.ai_wins += 1
        return self.winner

    @property
    def winner(self):
        """'player', 'ai', or None if the match is still in progress."""
        if self.player_wins >= self.wins_needed:
            return "player"
        if self.ai_wins >= self.wins_needed:
            return "ai"
        return None

    @property
    def over(self):
        return self.winner is not None

    def reset(self):
        self.player_wins = 0
        self.ai_wins = 0
