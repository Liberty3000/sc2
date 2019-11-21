import copy, os, tqdm
import numpy as np, torch as th
from pysc2.agents import base_agent
from sc2.controllers.actor_critic import ActorCritic
from sc2.policy.categorical import Categorical
from sc2.policy.spatial_categorical import SpatialCategorical
from sc2.memory.experience_replay import ExperienceReplay
from sc2.util import estimate_advantage, preprocess_observation, parameterize_action

class DDPG(base_agent.BaseAgent):
    def __init__(self, device=th.device('cpu'), model=ActorCritic(), *args, **kwargs):
        super().__init__()
        self.device = device
        self.actor, self.critic = model.actor, model.critic
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        self.actor_optim = th.optim.Adam(self.actor.parameters(), amsgrad=True, lr=2e-5, weight_decay=1e-5)
        self.critic_optim= th.optim.Adam(self.critic.parameters(),amsgrad=True, lr=2e-5, weight_decay=1e-5)
        self.lossfn = th.nn.MSELoss()
        self.memory = ExperienceReplay()
        self.policy = Categorical()
        self.spatial_policy = SpatialCategorical()

    def memorize(self, obs):
        g,l,n = preprocess_observation(obs)
        self.memory.retain(global_feats=self.global_feats,
                           local_feats =self.local_feats,
                           non_spatial_feats=self.non_spatial_feats,
                           action=self.action,
                           reward=obs.reward,
                           global_feats_prime=g,
                           local_feats_prime=l,
                           non_spatial_feats_prime=n,
                           terminal=obs.last())

    def forward(self, obs):
        output = preprocess_observation(obs)
        self.global_feats, self.local_feats, self.non_spatial_feats = output
        gfeat = self.global_feats.to(self.device)
        lfeat = self.local_feats.to(self.device)
        nfeat = self.non_spatial_feats.to(self.device)
        actor = self.actor.to(self.device)
        return actor(gfeat, lfeat, nfeat)

    def step(self, obs, recurrent=False):
        super().step(obs)

        if recurrent and obs.first():
            (self.cx, self.hx) = self.model.init_recurrence()

        contraints = np.asarray(obs.observation.available_actions)

        logits, spatial_args = self.forward(obs)

        self.action, self.logprob, self.entropy = self.policy(logits, contraints)

        self.action_args = self.spatial_policy(spatial_args)

        if recurrent:
            if obs.last():
                (self.cx, self.hx) = self.model.init_recurrence()
            else:
                self.cx, self.hx = self.cx.detach(), self.hx.detach()

        return parameterize_action(self.action, self.action_args, contraints)


    def update(self, obs, bsize=32, clip_param=2e-1, gamma=0.99, soft_tau=1e-2, *args, **kwargs):
        save_as = os.path.join(kwargs['worker'], 'training_stats.csv')
        if not os.path.isfile(save_as):
            open(save_as,'a+').write('value_loss,policy_loss,total_loss,transitions\n')

        losses = 0
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        for _ in tqdm.tqdm(range(bsize)):
            G,L,N,A,R,G_,L_,N_,T = self.memory.minibatch(bsize=1, device=self.device)

            logits,_ = self.actor(G,L,N)
            action,_,_ = self.policy(logits)
            value = self.critic(G,L,N, action)
            ploss = self.critic(G,L,N, action)
            ploss = -ploss.mean()

            logits,_ = self.target_actor(G_,L_,N_)
            aprime,_,_ = self.policy(logits)
            vtarget = self.target_critic(G_,L_,N_, aprime)
            vexpected = R + (1.0 - T) * gamma * vtarget
            vexpected = th.clamp(vexpected, -np.inf, np.inf)

            vloss = self.lossfn(value, vexpected.detach())

            self.actor_optim.zero_grad()
            ploss.backward()
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            vloss.backward()
            self.critic_optim.step()

            loss = (vloss + ploss).mean()
            losses += loss.item()

            args = (ploss.item(), vloss.item(), loss.item())
            with open(save_as,'a+') as f:
                f.write('{},{},{}\n'.format(*args))

        for target,param in zip(self.target_critic.parameters(),self.critic.parameters()):
            target.data.copy_(target.data * (1.0 - soft_tau) + param.data * soft_tau)

        for target,param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target.data.copy_(target.data * (1.0 - soft_tau) + param.data * soft_tau)

        return losses / bsize
