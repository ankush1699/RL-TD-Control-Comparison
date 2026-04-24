"""
Shared helpers for the TD agents. Kept separate so I'm not rewriting
eps-greedy selection in four places.
"""
import numpy as np


def eps_greedy_action(q_row, eps, rng):
    """Pick an action from eps-greedy over Q-values for a single state."""
    if rng.random() < eps:
        return rng.integers(len(q_row))
    # break ties randomly so we don't bias toward action 0
    best = np.flatnonzero(q_row == q_row.max())
    return rng.choice(best)


def eps_greedy_probs(q_row, eps):
    """
    Convert Q-values at a state into the eps-greedy action distribution.
    Used by Expected SARSA and RR-ES for the target expectation.
    Ties split the greedy mass evenly.
    """
    n = len(q_row)
    probs = np.full(n, eps / n)
    best = np.flatnonzero(q_row == q_row.max())
    probs[best] += (1.0 - eps) / len(best)
    return probs