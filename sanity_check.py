"""Quick check that all agents run without errors."""
import sys, os
sys.path.insert(0, '.')

from envs import CliffWalking, StochasticCliff
from agents import run_sarsa, run_q_learning, run_expected_sarsa, run_rr_expected_sarsa

env = CliffWalking()
Q, r = run_sarsa(env, 50, alpha=0.5, eps=0.1, seed=0)
print(f'SARSA ok, last-10 avg: {r[-10:].mean():.1f}')

env = CliffWalking()
Q, r = run_q_learning(env, 50, alpha=0.5, eps=0.1, seed=0)
print(f'Q-learning ok, last-10 avg: {r[-10:].mean():.1f}')

env = CliffWalking()
Q, r = run_expected_sarsa(env, 50, alpha=0.5, eps=0.1, seed=0)
print(f'Expected SARSA ok, last-10 avg: {r[-10:].mean():.1f}')

env = StochasticCliff(slip_prob=0.2, rng_seed=0)
Q, r = run_rr_expected_sarsa(env, 50, alpha=0.3, eps=0.1, delta=1.0, seed=0)
print(f'RR-ES on stochastic cliff ok, last-10 avg: {r[-10:].mean():.1f}')

print('\nAll agents working. Ready for Day 2.')