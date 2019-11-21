import torch as th
from sc2.policy import Policy
from sc2.util import compute_returns, mask_actions

class Categorical(Policy):
    def __init__(self):
        super().__init__()

    def __call__(self, logits, legal_actions=None):
        if legal_actions is not None:
            mask = mask_actions(legal_actions)
            if logits.is_cuda: mask = mask.cuda()
            logits = logits * mask
            logits = logits.squeeze()[legal_actions]

        probs = th.nn.Softmax(dim=-1)(logits)

        distr = th.distributions.Categorical(probs)

        action  = distr.sample()

        logprob = distr.log_prob(action)

        entropy = distr.entropy().mean()

        return action.item(), logprob, entropy
