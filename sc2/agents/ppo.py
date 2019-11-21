import os, numpy as np, torch as th
from pysc2.agents import base_agent
from sc2.controllers.atari_net import AtariNet
from sc2.policy.categorical import Categorical
from sc2.policy.spatial_categorical import SpatialCategorical
from sc2.memory.short_term import ShortTerm
from sc2.util import estimate_advantage, preprocess_observation, parameterize_action

class PPO(base_agent.BaseAgent):
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

        self.action, self.logprob, self.entropy = self.policy(logits, contraints)

        self.action_args = self.spatial_policy(spatial_args)

        if recurrent:
            if obs.last():
                (self.cx, self.hx) = self.model.init_recurrence()
            else:
                self.cx, self.hx = self.cx.detach(), self.hx.detach()

        return parameterize_action(self.action, self.action_args, contraints)

    def update(self, obs, clip_param=2e-1, *args, **kwargs):
        save_as = os.path.join(kwargs['worker'],'training_stats.csv')
        if not os.path.isfile(save_as):
            open(save_as,'a+').write('value_loss,policy_loss,total_loss\n')

        values, logprobs  = th.cat(self.memory.values), th.stack(self.memory.logprobs)
        rewards,terminals = self.memory.rewards, self.memory.terminals

        state_value,_,_ = self.forward(obs)

        returns = estimate_advantage(state_value, values, rewards, terminals)
        returns = th.cat(returns).detach()
        advantage = returns - values

        ratio = (self.logprob - logprobs).exp()
        surr1 = ratio * advantage
        surr2 = th.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

        ploss = -th.min(surr1, surr2).mean()
        vloss = (returns - values).pow(2).mean()

        loss = vloss + ploss

        args = (ploss.item(), vloss.item(), loss.item())
        with open(save_as,'a+') as f:
            f.write('{},{},{}\n'.format(*args))

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.memory.wipe()
        return loss.item()
