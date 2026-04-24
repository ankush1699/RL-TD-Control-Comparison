"""
Q-learning. Off-policy, uses max over next actions in the target.
Update: Q(s,a) += alpha * (r + gamma*max_a' Q(s',a') - Q(s,a))
"""
import numpy as np
from ._common import eps_greedy_action


def run_q_learning(env, n_episodes, alpha, eps, gamma=1.0, seed=0, max_steps=1000):
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

            target = r + gamma * Q[s_next].max()
            Q[s, a] += alpha * (target - Q[s, a])

            s = s_next
            steps += 1

        returns[ep] = total_r

    return Q, returns