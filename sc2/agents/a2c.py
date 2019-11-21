import os, numpy as np, torch as th
from pysc2.agents import base_agent
from sc2.controllers.atari_net import AtariNet
from sc2.policy.categorical import Categorical
from sc2.policy.spatial_categorical import SpatialCategorical
from sc2.memory.short_term import ShortTerm
from sc2.util import compute_returns, estimate_advantage
from sc2.util import preprocess_observation, parameterize_action

class A2C(base_agent.BaseAgent):
    def __init__(self, device=th.device('cpu'), model=AtariNet(), *args, **kwargs):
        super().__init__()
        self.device = device
        self.model = model
        self.optim = th.optim.Adam(self.model.parameters(), amsgrad=True, lr=2e-4)
        self.lossfn = th.nn.MSELoss()
        self.memory = ShortTerm()
        self.policy = Categorical()
        self.spatial_policy = SpatialCategorical()

    def forward(self, obs):
        output = preprocess_observation(obs)
        self.global_feats, self.local_feats, self.non_spatial_feats = output
        gfeat = self.global_feats.to(self.device)
        lfeat = self.local_feats.to(self.device)
        nfeat = self.non_spatial_feats.to(self.device)
        model = self.model.to(self.device)
        return model(gfeat, lfeat, nfeat)

    def memorize(self, obs):
        self.memory.retain(value=self.state_value, logprob=self.logprob, reward=obs.reward, terminal=obs.last())

    def step(self, obs, recurrent=False):
        super().step(obs)

        if recurrent and obs.first():
            (self.cx, self.hx) = self.model.init_recurrence()

        contraints = np.asarray(obs.observation.available_actions)

        self.state_value, logits, spatial_args = self.forward(obs)

        self.action, self.logprob, _ = self.policy(logits, contraints)

        self.action_args = self.spatial_policy(spatial_args)

        if recurrent:
            if obs.last():
                (self.cx, self.hx) = self.model.init_recurrence()
            else:
                self.cx, self.hx = self.cx.detach(), self.hx.detach()

        return parameterize_action(self.action, self.action_args, contraints)

    def update(self, obs, *args, **kwargs):
        save_as = os.path.join(kwargs['worker'],'training_stats.csv')
        if not os.path.isfile(save_as):
            open(save_as,'a+').write('value_loss,policy_loss,total_loss\n')

        values = th.stack(self.memory.values)
        logprobs = th.stack(self.memory.logprobs)
        entropies = th.stack(self.memory.entropies)
        rewards,terminals = self.memory.rewards, self.memory.terminals

        state_value,_,_ = self.forward(obs)

        returns = compute_returns(state_value, rewards, terminals)
        returns = th.cat(returns).detach()
        advantage = returns - values

        ploss = -(logprobs * advantage.detach()).mean()
        vloss = advantage.pow(2).mean()
        loss = vloss + ploss

        args = (vloss.item(), ploss.item(), loss.item())
        with open(save_as,'a+') as f:
            f.write('{},{},{}\n'.format(*args))

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.memory.wipe()
        return loss.item()
