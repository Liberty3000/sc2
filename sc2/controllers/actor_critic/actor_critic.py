import torch as th
from sc2.controllers.actor_critic.actor import Actor
from sc2.controllers.actor_critic.critic import Critic

class ActorCritic(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor, self.critic = Actor(), Critic()
    def __call__(self, *args):
        non_spatial_logits, (xlogits, ylogits) = self.actor(*args)
        state_value = self.critic(*args)
        return state_value, non_spatial_logits, (xlogits, ylogits)
