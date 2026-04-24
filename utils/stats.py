"""
Bootstrap statistics for comparing methods.
"""
import numpy as np


def bootstrap_mean_ci(values, n_boot=10000, alpha=0.05, seed=0):
    """Percentile-bootstrap (1-alpha) CI for the mean of values."""
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    lo = np.quantile(boot_means, alpha / 2)
    hi = np.quantile(boot_means, 1 - alpha / 2)
    return values.mean(), lo, hi


def bootstrap_diff_ci(a, b, n_boot=10000, alpha=0.05, seed=0):
    """
    Bootstrap CI and two-sided p-value for (mean_a - mean_b).
    Treats the two run sets as independent (not paired).
    """
    rng = np.random.default_rng(seed)
    n_a, n_b = len(a), len(b)
    idx_a = rng.integers(0, n_a, size=(n_boot, n_a))
    idx_b = rng.integers(0, n_b, size=(n_boot, n_b))
    boot_diffs = a[idx_a].mean(axis=1) - b[idx_b].mean(axis=1)
    diff_mean = a.mean() - b.mean()
    lo = np.quantile(boot_diffs, alpha / 2)
    hi = np.quantile(boot_diffs, 1 - alpha / 2)
    # two-sided p: proportion of resamples on the other side of 0
    if diff_mean >= 0:
        p = 2 * (boot_diffs < 0).mean()
    else:
        p = 2 * (boot_diffs > 0).mean()
    p = min(p, 1.0)
    return diff_mean, lo, hi, p


def pairwise_table(results_dict, metric_name='last30', n_boot=5000):
    """
    {method_name: array_of_per_run_values} -> prints pairwise comparison table.
    """
    methods = list(results_dict.keys())
    n = len(methods)
    print(f"\nPairwise comparisons on {metric_name} (bootstrap n={n_boot}):")
    print(f"{'Method A':<22}{'Method B':<22}{'diff (A-B)':<14}{'95% CI':<22}{'p':<8}")
    print('-' * 90)
    for i in range(n):
        for j in range(i + 1, n):
            a_vals = results_dict[methods[i]]
            b_vals = results_dict[methods[j]]
            diff, lo, hi, p = bootstrap_diff_ci(a_vals, b_vals, n_boot=n_boot)
            ci_str = f"[{lo:+.2f}, {hi:+.2f}]"
            sig = '**' if p < 0.01 else ('*' if p < 0.05 else '')
            print(f"{methods[i]:<22}{methods[j]:<22}{diff:+<14.2f}{ci_str:<22}{p:<6.3f}{sig}")
