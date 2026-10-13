# Comparing TD Control Methods on Cliff Walking

A reinforcement learning study that compares five temporal-difference (TD) control methods on four cliff walking gridworlds, with 500 runs per setting and statistical tests.

Course project for ECE 5474 (Reinforcement Learning), Virginia Tech, Spring 2026. The full write-up is in [Chaudhary_Ankush_P7_Project.pdf](Chaudhary_Ankush_P7_Project.pdf).

## Why it exists

Researchers have proposed many small changes to SARSA and Q-learning. They are usually tested on a single benchmark, so it is hard to tell which ones help in general. This project tests five methods across four environments that differ in layout and in randomness, and reports which differences are real.

The methods:

- SARSA
- Q-learning
- Expected SARSA
- Double Expected SARSA (Double ES)
- Regret-Regularized Expected SARSA (RR-ES), a variant I designed

## How it works

**Environments.** In every environment the agent gets -1 per step and -100 for stepping into a cliff cell (it then returns to the start). The episode ends at the goal. There are four combinations of layout and randomness:

| | Deterministic | Stochastic (20% chance the action is swapped for a random one) |
|---|---|---|
| Standard 4x12 grid with a cliff along the bottom | Det Cliff | Stoch Cliff |
| 7x12 corridor with a cliff on both sides | Corr Cliff | Stoch Corr Cliff |

**RR-ES.** Expected SARSA updates toward the average value of the next state under the agent's exploring policy. RR-ES drops any action whose value is more than a threshold `delta` below the best action before averaging. The idea was to stop rarely taken, very bad actions from dragging the estimate down.

**Protocol.** 100 episodes per run, 500 runs per setting, step size 0.3 and exploration rate 0.1 unless a sweep says otherwise. Scores use the average return over the last 30 episodes. Confidence intervals and pairwise tests use bootstrap resampling. All code is plain NumPy, with no RL libraries.

## How to run it

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python sanity_check.py   # quick check that all agents run
jupyter notebook
```

Then run notebooks 01 through 08 in order. Each has an `N_RUNS` setting near the top. Several notebooks are saved with `N_RUNS = 50` for quick pilots, and the main comparison and the sweeps over step size and exploration rate use 500. The report says the full 500-run campaign finished in under four hours on an Apple M1 MacBook.

Folders:

| Folder | Contents |
|---|---|
| `envs/` | The four environments |
| `agents/` | The five methods |
| `utils/` | Experiment runner, bootstrap statistics, behavior metrics |
| `notebooks/` | Eight notebooks that produce the figures |
| `results/data/` | Raw returns for each run (`.npz`) and behavior metrics (`.csv`) |
| `results/figures/` | Figures used in the report |

## Results

Average return over the last 30 episodes (higher is better), mean plus or minus standard error, 500 runs each. I recomputed these from `results/data/02_main_comparison.npz`. They match Table I of the report within rounding.

| Method | Det Cliff | Stoch Cliff | Corr Cliff | Stoch Corr Cliff |
|---|---|---|---|---|
| SARSA | -33.3 ± 0.3 | -101.3 ± 2.0 | -27.2 ± 0.3 | -68.9 ± 1.1 |
| Q-learning | -46.9 ± 0.5 | -87.9 ± 1.5 | -41.5 ± 0.4 | -64.0 ± 1.0 |
| Expected SARSA | **-26.2 ± 0.2** | **-87.3 ± 1.5** | **-26.5 ± 0.2** | -65.7 ± 1.0 |
| Double ES | -55.4 ± 0.8 | -121.8 ± 2.1 | -47.7 ± 0.6 | -116.6 ± 1.7 |
| RR-ES (delta = 1) | -48.2 ± 0.5 | -87.6 ± 1.6 | -42.5 ± 0.4 | -63.6 ± 1.0 |

What the report finds:

- **Expected SARSA** is the best method, or statistically tied for best, in all four environments.
- **RR-ES** does not beat Expected SARSA in any environment, at any delta tested. It is statistically indistinguishable from Q-learning in all four. This is a negative result.
- **Double ES** is the worst in every environment. On the stochastic corridor cliff it is 47 to 53 return points behind the others (p < 0.001). The report ties this to the short 100-episode training horizon.
- **Safety.** Under stochastic transitions, Expected SARSA falls off the cliff least often. Double ES falls 2 to 5 times as often (for example 0.32 against 0.06 falls per episode on the stochastic corridor cliff).

Limits, as stated in the report: tabular problems only, short training, small grids, one family of environments. The results may not carry over to function approximation or long-horizon tasks.

## Tech stack

Python, NumPy, pandas, matplotlib, tqdm, Jupyter.

## Author

Ankush Kailasrao Chaudhary, Virginia Tech (ankushchaudhary@vt.edu)
