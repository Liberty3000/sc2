import copy, os, tqdm
import numpy as np, torch as th
from pysc2.agents import base_agent

class Agent(base_agent.BaseAgent):
    def __init__(self, lock=None, counter=None, device=th.device('cpu'), *args, **kwargs):
        super().__init__()
        self.device, self.counter, self.lock = device, counter, lock
        self.model = None
        self.memory = None
        self.optim = None
        self.lossfn = None
        self.policy, self.spatial_policy = None, None

    def forward(self, obs, recurrent=False):
        output = preprocess_observation(obs)
        self.global_feats, self.local_feats, self.non_spatial_feats = output
        gfeat = self.global_feats.to(self.device)
        lfeat = self.local_feats.to(self.device)
        nfeat = self.non_spatial_feats.to(self.device)
        model = self.model.to(self.device)

        if recurrent:
            return model(gfeat, lfeat, nfeat, (self.cx,self.hx))
        else:
            return model(gfeat, lfeat, nfeat)

    def step(self, obs, recurrent=False):
        super().step(obs)

        if recurrent and obs.first():
            (self.cx, self.hx) = self.model.init_recurrence()

        contraints = np.asarray(obs.observation.available_actions)

        self.state_value, logits, spatial_args = self.forward(obs, recurrent=recurrent)

        self.action, self.logprob, self.entropy = self.policy(logits, contraints)

        self.action_args = self.spatial_policy(spatial_args)

        if recurrent:
            if obs.last():
                (self.cx, self.hx) = self.model.init_recurrence()
            else:
                self.cx, self.hx = self.cx.detach(), self.hx.detach()

        return parameterize_action(self.action, self.action_args, contraints)

    def memorize(self, obs):
        raise NotImplementedError

    def update(self, obs, *args, **kwargs):
        raise NotImplementedError
