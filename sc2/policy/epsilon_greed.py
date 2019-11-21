import matplotlib.pyplot as mp, numpy as np
from sc2.policy import Policy
from sc2.util import mask_actions

class EpsilonGreed(Policy):
    def __init__(self, epsilon_min=1e-2, epsilon_init=1.0, epsilon_decay=2**12):
        super().__init__()
        self.epsilon = self.epsilon_init = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.greedy = False
        self.calls = 0

    def decay_epsilon(self, itr):
        decay = np.exp(-1 * itr / self.epsilon_decay)
        self.epsilon = self.epsilon_min + (self.epsilon_init - self.epsilon_min) * decay
        return self.epsilon

    def __call__(self, non_spatial_logits, legal_actions=None):
        self.calls += 1

        if legal_actions is not None:
            mask = mask_actions(legal_actions)
            if non_spatial_logits.is_cuda: mask = mask.cuda()
            non_spatial_logits = non_spatial_logits.cpu().squeeze().detach().numpy()
            non_spatial_logits = non_spatial_logits[list(legal_actions)]

        if self.greedy or np.random.random() > self.epsilon:
            action = non_spatial_logits.argmax().item()
        else:
            if legal_actions is None:
                action = np.random.choice(range(non_spatial_logits.size(-1)))
            else:
                action = np.random.choice(range(len(legal_actions)))

            self.decay_epsilon(self.calls)

        return action
