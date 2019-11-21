import copy, os, tqdm
from torch.autograd import Variable
import numpy as np, torch as th
from pysc2.agents import base_agent
from sc2.controllers.fully_conv import FullyConv
from sc2.policy.epsilon_greed import EpsilonGreed
from sc2.policy.spatial_epsilon_greed import SpatialEpsilonGreed
from sc2.memory.experience_replay import ExperienceReplay
from sc2.util import compute_returns, estimate_advantage
from sc2.util import preprocess_observation, parameterize_action

class DQN(base_agent.BaseAgent):
    def __init__(self, target_network=False, device=th.device('cpu'), model=FullyConv(), *args, **kwargs):
        super().__init__()
        self.device = device
        self.model = model
        self.optim = th.optim.Adam(self.model.parameters())
        self.lossfn = th.nn.SmoothL1Loss()
        self.memory = ExperienceReplay()
        self.policy = EpsilonGreed()
        self.spatial_policy = SpatialEpsilonGreed()
        self.target_network = copy.deepcopy(self.model) if target_network else None

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

        self.action = self.policy(logits, contraints)

        self.action_args = self.spatial_policy(spatial_args)

        if recurrent:
            if obs.last():
                (self.cx, self.hx) = self.model.init_recurrence()
            else:
                self.cx, self.hx = self.cx.detach(), self.hx.detach()

        return parameterize_action(self.action, self.action_args, contraints)

    def update(self, obs, bsize=1, gamma=0.99, *args, **kwargs):
        save_as = os.path.join(kwargs['worker'],'training_stats.csv')
        if not os.path.isfile(save_as):
            banner = 'Qvalue,Qprime,Qexpect,loss\n'
            open(save_as,'a+').write(banner)

        losses = 0
        self.optim.zero_grad()
        for _ in tqdm.tqdm(range(bsize)):
            G,L,N,A,R,G_,L_,N_,T = self.memory.minibatch(bsize=1, device=self.device)

            _,Qvalue,_ = self.model( G, L, N)
            Qvalue = Qvalue.unsqueeze(0)
            _,Qprime,_ = self.model(G_,L_,N_)
            Qprime = Qprime.unsqueeze(0)

            Qvalue = Qvalue.gather(1, A.unsqueeze(1).long()).squeeze(1)
            Qprime = Qprime.gather(1, th.max(Qprime, 1)[1].unsqueeze(1)).squeeze(1)

            Qvalue = Variable(Qvalue, requires_grad=True)
            Qprime = Variable(Qprime, requires_grad=False)

            Qexpect = R + gamma * Qprime * (1 - T)

            loss = self.lossfn(Qvalue, Qexpect)

            loss.backward()
            losses += loss.item()

            args = (Qvalue.item(), Qprime.item(), Qexpect.item(), loss.item())
            with open(save_as, 'a+') as f:
                f.write('{},{},{},{}\n'.format(*args))

        self.optim.step()
        return losses / bsize
