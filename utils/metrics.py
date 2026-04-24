"""
Policy-behavior metrics beyond average return.
"""
import inspect
import numpy as np


def rollout_metrics(env_fn, Q, n_rollouts=50, max_steps=500,
                     cliff_reward=-100.0, seed=0):
    """
    Evaluate a GREEDY policy wrt Q for n_rollouts episodes.
    Returns per-rollout aggregates (mean/std) of steps, falls, returns.
    """
    steps_list = []
    falls_list = []
    returns_list = []
    successes = 0

    sig = inspect.signature(env_fn)
    env_takes_seed = 'seed' in sig.parameters

    for k in range(n_rollouts):
        env = env_fn(seed=seed + k) if env_takes_seed else env_fn()
        s = env.reset()
        total_r = 0.0
        falls = 0
        steps = 0
        done = False

        while not done and steps < max_steps:
            a = int(np.argmax(Q[s]))
            s_next, r, done = env.step(a)
            total_r += r
            if r == cliff_reward:
                falls += 1
            s = s_next
            steps += 1

        steps_list.append(steps)
        falls_list.append(falls)
        returns_list.append(total_r)
        if done:
            successes += 1

    return {
        'steps_mean':   float(np.mean(steps_list)),
        'steps_std':    float(np.std(steps_list)),
        'falls_mean':   float(np.mean(falls_list)),
        'falls_std':    float(np.std(falls_list)),
        'return_mean':  float(np.mean(returns_list)),
        'return_std':   float(np.std(returns_list)),
        'success_rate': successes / n_rollouts,
    }


def regret_to_convergence(returns, threshold_frac=0.9, window=10):
    """
    Given a 1D array of per-episode returns, estimate:
      - episode_reached: when smoothed return first hits threshold_frac of final
      - regret: cumulative gap (final_perf - return_t) up to that episode
    """
    returns = np.asarray(returns)
    final_perf = returns[-window:].mean()
    threshold = threshold_frac * final_perf
    smoothed = np.convolve(returns, np.ones(window) / window, mode='same')
    hit = np.where(smoothed >= threshold)[0]
    if len(hit) == 0:
        ep = len(returns) - 1
    else:
        ep = int(hit[0])
    regret = float(np.sum(final_perf - returns[:ep + 1]))
    return {'episode_reached': ep, 'regret': regret, 'final_perf': float(final_perf)}
