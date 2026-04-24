"""
Corridor cliff walking. Narrow passage with cliffs on BOTH sides of the route.

For rows=7, cols=12:
    Row 0: cliff from col 1 to col cols-2 (top cliff)
    Rows 1..5: open corridor
    Row 6: cliff from col 1 to col cols-2 (bottom cliff)
    Start: (3, 0). Goal: (3, cols-1).

Stepping into a cliff cell: -100 reward, reset to start (episode continues).
All other steps: -1 reward. Episode ends at goal.
"""
import numpy as np
from .cliff_walking import CliffWalking, UP, DOWN, LEFT, RIGHT, N_ACTIONS


class CorridorCliff(CliffWalking):
    def __init__(self, rows=7, cols=12):
        super().__init__(rows=rows, cols=cols)
        mid_row = rows // 2
        self.start = (mid_row, 0)
        self.goal  = (mid_row, cols - 1)

        self.cliff = set()
        for c in range(1, cols - 1):
            self.cliff.add((0, c))
            self.cliff.add((rows - 1, c))

        self.state = self.start


class StochasticCorridorCliff(CorridorCliff):
    """Corridor cliff with action-slip stochasticity."""
    def __init__(self, rows=7, cols=12, slip_prob=0.2, rng_seed=None):
        super().__init__(rows=rows, cols=cols)
        self.slip_prob = slip_prob
        self.rng = np.random.default_rng(rng_seed)

    def step(self, action):
        if self.rng.random() < self.slip_prob:
            others = [a for a in range(N_ACTIONS) if a != action]
            action = self.rng.choice(others)
        return super().step(action)
