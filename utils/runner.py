"""
Runs an agent many times with different seeds and averages the
per-episode returns. This is what van Seijen Fig 2 does -- average
over many independent runs so the curves are smooth.
"""
import inspect
import numpy as np

try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def multi_seed_run(agent_fn, env_fn, n_runs, n_episodes,
                   progress=False, desc=None, **agent_kwargs):
    """
    agent_fn: callable like run_sarsa(env, n_episodes, ..., seed=k)
    env_fn:   callable that returns a fresh env instance. If it accepts
              a `seed` kwarg we pass the run index so env randomness
              (e.g. stochastic cliff slip) also varies across runs.
    progress: show a tqdm bar over runs (requires tqdm installed).
    desc:     description string for the progress bar.
    Returns an array of shape (n_runs, n_episodes).
    """
    sig = inspect.signature(env_fn)
    env_takes_seed = "seed" in sig.parameters

    all_returns = np.zeros((n_runs, n_episodes))
    iterator = range(n_runs)
    if progress and _HAS_TQDM:
        iterator = tqdm(iterator, desc=desc, leave=False)

    for k in iterator:
        env = env_fn(seed=k) if env_takes_seed else env_fn()
        _, returns = agent_fn(env, n_episodes, seed=k, **agent_kwargs)
        all_returns[k] = returns
    return all_returns


def smooth(x, window=10):
    """Simple moving average for plotting. Edges get shorter windows."""
    if window <= 1:
        return x
    out = np.empty_like(x, dtype=float)
    n = len(x)
    for i in range(n):
        lo = max(0, i - window // 2)
        hi = min(n, i + window // 2 + 1)
        out[i] = x[lo:hi].mean()
    return out
