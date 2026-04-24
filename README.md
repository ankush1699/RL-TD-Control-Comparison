# A Systematic Comparison of Temporal-Difference Control Methods on Cliff Walking Variants

Course project for ECE 5474 (Reinforcement Learning), Virginia Tech, Spring 2026.

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Then run notebooks in order 01 through 08.

## Structure

- `envs/` — four cliff walking environments (deterministic, stochastic, corridor, stochastic corridor)
- `agents/` — five TD control methods (SARSA, Q-learning, Expected SARSA, Double Expected SARSA, RR-ES)
- `utils/` — experiment runner, bootstrap statistics, behavioral metrics
- `notebooks/` — 8 notebooks that generate all paper figures
- `results/data/` — raw per-run returns (.npz format)
- `results/figures/` — figures used in the paper

## Reproducing the paper

Each notebook has an `N_RUNS` constant at the top. Set to 50 for a quick pilot
(~6 minutes total) or 500 for publication-quality results (~3 hours).

## Author

Ankush Kailasrao Chaudhary, Virginia Tech (ankushchaudhary@vt.edu)