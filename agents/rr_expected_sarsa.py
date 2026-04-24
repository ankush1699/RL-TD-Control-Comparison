"""
Regret-Regularized Expected SARSA (RR-ES).

Same as Expected SARSA but the target expectation is taken over a
pessimism-filtered action distribution: we zero out any action a'
whose Q(s',a') is more than delta below V(s') = max_a Q(s',a),
then renormalize the remaining probabilities.

delta = inf   -> recovers standard Expected SARSA
delta = 0     -> only greedy action(s) count, approximates Q-learning's target
                 (but note the behavior policy is still eps-greedy, so it's
                 not exactly Q-learning even at delta=0)

Intuition: on tasks with catastrophic actions (like cliff walking), the
eps-greedy expectation drags Q-estimates down via low-value actions
that the agent would rarely want to take anyway. Filtering them out
lets the target reflect "what I'll probably actually do" better.
"""
import numpy as np
from ._common import eps_greedy_action, eps_greedy_probs


def rr_target_probs(q_row, eps, delta):
    """
    Start from the eps-greedy distribution, then zero out any action
    with Q more than delta below max, and renormalize.
    """
    probs = eps_greedy_probs(q_row, eps)
    v = q_row.max()
    # mask keeps actions within delta of the best
    mask = (q_row >= v - delta)
    probs = probs * mask
    total = probs.sum()
    if total <= 0:
        # shouldn't happen since max action is always in the mask,
        # but guard against numerical weirdness
        probs = np.zeros_like(q_row)
        best = np.flatnonzero(q_row == v)
        probs[best] = 1.0 / len(best)
        return probs
    return probs / total


def run_rr_expected_sarsa(env, n_episodes, alpha, eps, delta,
                          gamma=1.0, seed=0, max_steps=1000):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    returns = np.zeros(n_episodes)

    for ep in range(n_episodes):
        s = env.reset()
        total_r = 0.0
        done = False
        steps = 0

        while not done and steps < max_steps:
            a = eps_greedy_action(Q[s], eps, rng)
            s_next, r, done = env.step(a)
            total_r += r

            if done:
                Q[s, a] += alpha * (r - Q[s, a])
                break

            probs = rr_target_probs(Q[s_next], eps, delta)
            expected_v = np.dot(probs, Q[s_next])
            target = r + gamma * expected_v
            Q[s, a] += alpha * (target - Q[s, a])

            s = s_next
            steps += 1

        returns[ep] = total_r

    return Q, returns