import copy, os, numpy as np, torch as th
from pysc2.agents import base_agent
from sc2.controllers.recurrent_ac.fused_lstm import FusedLSTM
from sc2.policy.categorical import Categorical
from sc2.policy.spatial_categorical import SpatialCategorical
from sc2.memory.short_term import ShortTerm
from sc2.optimizer.shared_adam import SharedAdam
from sc2.util import compute_returns, estimate_advantage
from sc2.util import preprocess_observation, parameterize_action

class A3C(base_agent.BaseAgent):
    def __init__(self, lock, counter, device=th.device('cpu'), model=FusedLSTM(), *args, **kwargs):
        super().__init__()
        self.device = device
        self.lock, self.counter = lock, counter

        self.model = model
        self.shared_model = copy.deepcopy(self.model)
        self.shared_model.share_memory()

        self.optim = SharedAdam(self.model.parameters(), lr=2e-4)
        self.lossfn = th.nn.MSELoss()
        self.memory = ShortTerm()
        self.policy = Categorical()
        self.spatial_policy = SpatialCategorical()

    def forward(self, obs):
        self.model.load_state_dict(self.shared_model.state_dict())
        output = preprocess_observation(obs)
        self.global_feats, self.local_feats, self.non_spatial_feats = output
        gfeat = self.global_feats.to(self.device)
        lfeat = self.local_feats.to(self.device)
        nfeat = self.non_spatial_feats.to(self.device)
        model = self.model.to(self.device)
        return model(gfeat, lfeat, nfeat, (self.cx,self.hx))

    def memorize(self, obs):
        self.memory.retain(value=self.state_value, logprob=self.logprob,
        entropy=self.entropy, reward=obs.reward, terminal=obs.last())

    def step(self, obs, recurrent=True):
        super().step(obs)

        if recurrent and obs.first():
            (self.cx, self.hx) = self.model.init_recurrence()

        with self.lock:
            self.counter.value += 1

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

    def ensure_shared_grads(self):
        for param, shared_param in zip(self.model.parameters(), self.shared_model.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad

    def update(self, obs, max_gradnorm=50, entropy_coeff=1e-3, *args, **kwargs):
        save_as = os.path.join(kwargs['worker'],'training_stats.csv')
        if not os.path.isfile(save_as):
            banner = 'value_loss,policy_loss,entropy_loss,total_loss,transitions\n'

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
        eloss = entropy_coeff - entropies
        loss = (vloss + ploss - eloss).mean()

        args = (vloss.mean(), ploss.mean(), eloss.mean(), loss, len(rewards))
        with open(save_as, 'a+') as f:
            f.write('{},{},{},{},{}\n'.format(*args))

        self.optim.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.model.parameters(), max_gradnorm)

        self.ensure_shared_grads()
        self.optim.step()

        self.memory.wipe()

        string = kwargs['worker'][:kwargs['worker'].rindex('/')]
        save_as = os.path.join(string, 'shared_model.th')
        th.save(self.shared_model.state_dict(), save_as)
        return loss.item()
