"""
Stochastic cliff walking. Same layout as standard cliff, but with
probability `slip_prob` the agent's action is replaced by a uniformly
random one of the other three actions. This is the same noise model
van Seijen et al. used for their stochastic windy gridworld (20%).

The env noise matters because eps-greedy ES's expectation includes
all actions weighted by the *policy*, but not the env's transition
noise. So when env noise is high, ES is still estimating "my policy's
expected value", which is not the same as "what will actually happen".
"""
import numpy as np
from .cliff_walking import CliffWalking, UP, DOWN, LEFT, RIGHT, N_ACTIONS


class StochasticCliff(CliffWalking):
    def __init__(self, rows=4, cols=12, slip_prob=0.2, rng_seed=None):
        super().__init__(rows=rows, cols=cols)
        self.slip_prob = slip_prob
        # each env instance gets its own RNG so multi-seed runs are independent
        self.rng = np.random.default_rng(rng_seed)

    def step(self, action):
        # with prob slip_prob, swap to a random different action
        if self.rng.random() < self.slip_prob:
            others = [a for a in range(N_ACTIONS) if a != action]
            action = self.rng.choice(others)
        return super().step(action)