import copy, os, tqdm
from torch.autograd import Variable
import numpy as np, torch as th
from pysc2.agents import base_agent
from sc2.agents.dqn import DQN
from sc2.controllers.fully_conv import FullyConv
from sc2.policy.epsilon_greed import EpsilonGreed
from sc2.policy.spatial_epsilon_greed import SpatialEpsilonGreed
from sc2.memory.experience_replay import ExperienceReplay
from sc2.util import compute_returns, estimate_advantage
from sc2.util import preprocess_observation, parameterize_action

class SARSA(DQN):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)

    def update(self, obs, obs_, bsize=1, gamma=0.99, *args, **kwargs):
        save_as = os.path.join(kwargs['worker'],'training_stats.csv')
        if not os.path.isfile(save_as):
            banner = 'Qvalue,Qprime,Qexpect,loss\n'
            open(save_as,'a+').write(banner)

        self.optim.zero_grad()

        self.step(obs)
        A = self.action
        self.step(obs_)
        A_= self.action

        Qvalue = Qvalue.gather(1, A.unsqueeze(1).long()).squeeze(1)
        Qprime = Qvalue.gather(1,A_.unsqueeze(1).long()).squeeze(1)

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
        return loss.item()
