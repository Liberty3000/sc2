import torch as th
from sc2.policy import Policy
from sc2.util import compute_returns, mask_actions

class Gaussian(Policy):
    def __init__(self):
        super().__init__()

    def __call__(self, mean, stdv):
        distr = th.distributions.Normal(mean, stdv)

        action  = distr.sample()

        logprob = distr.log_prob(action)

        entropy = distr.entropy().mean()

        return action.item(), logprob, entropy
