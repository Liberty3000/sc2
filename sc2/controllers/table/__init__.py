import traceback
import numpy as np, pandas as pd, torch as th
from sc2.policy.epsilon_greed import EpsilonGreed

class Table:
    def __init__(self, action_space, policy=EpsilonGreed()):
        self.action_space = action_space
        self.df = pd.DataFrame(columns=action_space)
        self.policy = policy

    def assess_novelty(self, state):
        if str(state) not in self.df.index:
            action_values = pd.Series([0] * len(self.action_space),
                                      index=self.df.columns,
                                      name=str(state))
            self.df = self.df.append(action_values)

    def __call__(self, state, available_actions=None):
        self.assess_novelty(state)

        try:
            arr = self.df.ix[str(state),:].values.astype(np.float32)
            action = self.policy(th.from_numpy(arr))
        except:
            print(traceback.format_exc())
            action = 0

        return action

    def update(self, state, action, reward, sprime, terminal, on_policy=True, gamma=0.99, lrate=1e-2):
        self.assess_novelty(state)
        self.assess_novelty(sprime)

        qvalue = self.df.ix[str(state), action]

        if on_policy:
            logits = self.df.ix[str(sprime),:].values.astype(np.float32)
            aprime = self.policy(th.from_numpy(logits))
            qprime = self.df.ix[str(sprime), aprime]
        if not on_policy:
            qprime = qvalue = self.df.ix[str(sprime),:].max()

        qtarget = reward if terminal else reward + gamma * qprime

        loss = (qtarget - qvalue)

        self.df.ix[str(state), action] = qvalue + (lrate * loss)

        return float(loss)
