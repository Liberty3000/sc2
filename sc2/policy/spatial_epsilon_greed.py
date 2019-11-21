import numpy as np
from sc2.policy import Policy

class SpatialEpsilonGreed(Policy):
    def __init__(self, spatial_extent=(64,64), epsilon_min=0.05, epsilon_init=1.0, epsilon_decay=2**12):
        super().__init__()
        self.spatial_extent = spatial_extent
        self.epsilon = self.epsilon_init = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.greedy = False
        self.calls = 0

    def decay_epsilon(self, itr):
        decay = np.exp(-1 * itr / self.epsilon_decay)
        self.epsilon = self.epsilon_min + (self.epsilon_init - self.epsilon_min) * decay
        return self.epsilon

    def __call__(self, logits, legal_actions=None):
        self.calls += 1

        if self.greedy or np.random.random() > self.epsilon:
            heatmap_amax = logits.view(-1).argmax().cpu().detach().numpy()
            action_args = np.unravel_index(heatmap_amax, self.spatial_extent)
        else:
            xarg = np.random.randint(0, self.spatial_extent[0])
            yarg = np.random.randint(0, self.spatial_extent[1])
            action_args = [xarg, yarg]
            self.decay_epsilon(self.calls)

        return action_args
