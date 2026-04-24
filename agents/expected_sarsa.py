"""
Expected SARSA from van Seijen et al. 2009.
On-policy but uses the expectation over the next action distribution
instead of a single sample. Lower variance than plain SARSA.

Update: Q(s,a) += alpha * (r + gamma * sum_a' pi(a'|s') Q(s',a') - Q(s,a))
"""
import numpy as np
from ._common import eps_greedy_action, eps_greedy_probs


def run_expected_sarsa(env, n_episodes, alpha, eps, gamma=1.0, seed=0, max_steps=1000):
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

            probs = eps_greedy_probs(Q[s_next], eps)
            expected_v = np.dot(probs, Q[s_next])
            target = r + gamma * expected_v
            Q[s, a] += alpha * (target - Q[s, a])

            s = s_next
            steps += 1

        returns[ep] = total_r

    return Q, returns