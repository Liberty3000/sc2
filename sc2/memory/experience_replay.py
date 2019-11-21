from collections import deque
import random, numpy as np, torch as th
from torch.autograd import Variable

class ExperienceReplay(object):
    def __init__(self, capacity=2**11):
        self.capacity = capacity
        self.wipe()

    def wipe(self):
        self.actions = deque(maxlen=self.capacity)
        self.rewards = deque(maxlen=self.capacity)
        self.terminals = deque(maxlen=self.capacity)
        self.global_feats = deque(maxlen=self.capacity)
        self.local_feats = deque(maxlen=self.capacity)
        self.non_spatial_feats = deque(maxlen=self.capacity)
        self.global_feats_prime = deque(maxlen=self.capacity)
        self.local_feats_prime = deque(maxlen=self.capacity)
        self.non_spatial_feats_prime = deque(maxlen=self.capacity)

    def retain(self, global_feats, local_feats, non_spatial_feats,
               action, reward,
               global_feats_prime, local_feats_prime, non_spatial_feats_prime,
               terminal):

        self.global_feats.append(global_feats)
        self.local_feats.append(local_feats)
        self.non_spatial_feats.append(non_spatial_feats)

        self.actions.append(np.array(action))
        self.rewards.append(np.array(reward))
        self.terminals.append(np.array(int(terminal)))

        self.global_feats_prime.append(global_feats_prime)
        self.local_feats_prime.append(local_feats_prime)
        self.non_spatial_feats_prime.append(non_spatial_feats_prime)

    def minibatch(self, bsize=1, device=th.device('cpu')):
        G,L,N,A,R,G_,L_,N_,T = self.sample(bsize)

        G = Variable(th.from_numpy(G).float()).to(device)
        L = Variable(th.from_numpy(L).float()).to(device)
        N = Variable(th.from_numpy(N).float()).to(device)

        A = Variable(th.from_numpy(A).float()).to(device)
        R = Variable(th.from_numpy(R).float()).to(device)

        G_= Variable(th.from_numpy(G_).float()).to(device)
        L_= Variable(th.from_numpy(L_).float()).to(device)
        N_= Variable(th.from_numpy(N_).float()).to(device)

        T = Variable(th.from_numpy(T).float()).to(device)

        return G,L,N,A,R,G_,L_,N_,T

    def sample(self, bsize=1):
        idxs = [np.random.randint(0, len(self.global_feats)) for _ in range(bsize)]

        global_feats = np.concatenate(self.global_feats)
        local_feats = np.concatenate(self.local_feats)
        non_spatial_feats = np.concatenate(self.non_spatial_feats)
        actions = np.concatenate(np.asarray(self.actions).reshape(-1,1))
        rewards = np.concatenate(np.asarray(self.rewards).reshape(-1,1))
        terminals = np.concatenate(np.asarray(self.terminals).reshape(-1,1))
        global_feats_prime = np.concatenate(self.global_feats_prime)
        local_feats_prime = np.concatenate(self.local_feats_prime)
        non_spatial_feats_prime = np.concatenate(self.non_spatial_feats_prime)

        G = global_feats[idxs]
        L = local_feats[idxs]
        N = non_spatial_feats[idxs]

        A = actions[idxs]
        R = rewards[idxs]
        T = terminals[idxs]

        G_= global_feats_prime[idxs]
        L_= local_feats_prime[idxs]
        N_= non_spatial_feats_prime[idxs]

        return G,L,N,A,R,G_,L_,N_,T

    def __len__(self):
        assert len(self.global_feats) == \
        len(self.local_feats) == \
        len(self.non_spatial_feats) == \
        len(self.actions) == \
        len(self.rewards) == \
        len(self.global_feats_prime) == \
        len(self.local_feats_prime) == \
        len(self.non_spatial_feats_prime)
        return len(self.global_feats)
