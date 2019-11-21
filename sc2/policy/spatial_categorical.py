import numpy as np, torch as th
from sc2.policy import Policy
from sc2.util import compute_returns, mask_actions

class SpatialCategorical(Policy):
    def __init__(self, spatial_extent=(64,64)):
        super().__init__()
        self.spatial_extent = spatial_extent

    def __call__(self, logits):
        if isinstance(logits, tuple) or isinstance(logits, list):
            xdistr = th.distributions.Categorical(logits[0])
            x = xdistr.sample()
            ydistr = th.distributions.Categorical(logits[1])
            y = ydistr.sample()
            return (x,y)

        probs = th.nn.Softmax(dim=-1)(logits.view(-1))

        distr = th.distributions.Categorical(probs)

        action  = distr.sample()

        logprob = distr.log_prob(action)

        entropy = distr.entropy().mean()

        action = np.unravel_index(action.numpy(), self.spatial_extent)

        return action
