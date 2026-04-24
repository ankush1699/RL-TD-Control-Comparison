"""
Standard cliff walking from Sutton & Barto, Example 6.6.
Grid is 4 rows x 12 cols, start at bottom-left, goal at bottom-right,
cliff across the bottom row in between. Step reward -1, cliff -100
(and you snap back to start). Episode ends when you reach the goal.

Coordinates: (row, col). Row 0 is top, row 3 is bottom.
Actions: 0=up, 1=right, 2=down, 3=left.
"""
import numpy as np


UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
N_ACTIONS = 4


class CliffWalking:
    def __init__(self, rows=4, cols=12):
        self.rows = rows
        self.cols = cols
        self.start = (rows - 1, 0)
        self.goal = (rows - 1, cols - 1)
        # cliff cells: bottom row, columns 1..cols-2
        self.cliff = {(rows - 1, c) for c in range(1, cols - 1)}
        self.n_states = rows * cols
        self.n_actions = N_ACTIONS
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self._to_idx(self.state)

    def step(self, action):
        r, c = self.state
        if action == UP:
            nr, nc = max(r - 1, 0), c
        elif action == DOWN:
            nr, nc = min(r + 1, self.rows - 1), c
        elif action == LEFT:
            nr, nc = r, max(c - 1, 0)
        elif action == RIGHT:
            nr, nc = r, min(c + 1, self.cols - 1)
        else:
            raise ValueError(f"bad action {action}")

        next_pos = (nr, nc)

        if next_pos in self.cliff:
            # fell off — big penalty, reset to start, episode continues
            self.state = self.start
            return self._to_idx(self.state), -100.0, False

        self.state = next_pos
        reward = -1.0
        done = (next_pos == self.goal)
        return self._to_idx(self.state), reward, done

    def _to_idx(self, pos):
        r, c = pos
        return r * self.cols + c

    def idx_to_pos(self, idx):
        return (idx // self.cols, idx % self.cols)