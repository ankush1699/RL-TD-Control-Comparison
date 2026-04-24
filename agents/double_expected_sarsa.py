"""
Double Expected SARSA.

Extends Expected SARSA with the double-estimator trick from van Hasselt (2010):
maintain two Q-tables, Q_a and Q_b. On each step flip a coin; the chosen
table gets updated, but the target's action-probabilities come from THAT
table while the Q-values being averaged come from the OTHER table.

The decoupling reduces maximization bias -- useful when environment
stochasticity or high variance Q-estimates cause overestimation of
certain actions.

Action selection uses the average of Q_a and Q_b (for stability).
"""
import numpy as np
from ._common import eps_greedy_probs


def run_double_expected_sarsa(env, n_episodes, alpha, eps,
                               gamma=1.0, seed=0, max_steps=1000):
    rng = np.random.default_rng(seed)
    Q_a = np.zeros((env.n_states, env.n_actions))
    Q_b = np.zeros((env.n_states, env.n_actions))
    returns = np.zeros(n_episodes)

    def pick_action(s):
        # eps-greedy wrt the average of the two tables
        if rng.random() < eps:
            return rng.integers(env.n_actions)
        q_avg = (Q_a[s] + Q_b[s]) / 2.0
        best = np.flatnonzero(q_avg == q_avg.max())
        return rng.choice(best)

    for ep in range(n_episodes):
        s = env.reset()
        total_r = 0.0
        done = False
        steps = 0

        while not done and steps < max_steps:
            a = pick_action(s)
            s_next, r, done = env.step(a)
            total_r += r

            if done:
                if rng.random() < 0.5:
                    Q_a[s, a] += alpha * (r - Q_a[s, a])
                else:
                    Q_b[s, a] += alpha * (r - Q_b[s, a])
                break

            update_a = (rng.random() < 0.5)
            if update_a:
                probs = eps_greedy_probs(Q_a[s_next], eps)
                expected_v = np.dot(probs, Q_b[s_next])
                target = r + gamma * expected_v
                Q_a[s, a] += alpha * (target - Q_a[s, a])
            else:
                probs = eps_greedy_probs(Q_b[s_next], eps)
                expected_v = np.dot(probs, Q_a[s_next])
                target = r + gamma * expected_v
                Q_b[s, a] += alpha * (target - Q_b[s, a])

            s = s_next
            steps += 1

        returns[ep] = total_r

    Q = (Q_a + Q_b) / 2.0
    return Q, returns
