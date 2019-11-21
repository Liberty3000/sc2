import collections, numpy as np

class ShortTerm(object):
    def __init__(self):
        self.wipe()
    def retain(self, value=None, logprob=None, reward=None, entropy=None, terminal=None, action=None):
        self.actions   += [action]
        self.entropies += [entropy]
        self.values    += [value]
        self.logprobs  += [logprob]
        self.rewards   += [reward]
        self.terminals += [terminal]
    def __len__(self):
        return len(self.values)
    def wipe(self):
        self.actions = []
        self.values = []
        self.logprobs = []
        self.rewards = []
        self.entropies = []
        self.terminals = []
