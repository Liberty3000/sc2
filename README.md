# sc2

sc2 is a library for simulating and training agents in the StarCraft II learning environment.

To install the package, run the following commands:
```
git clone git@github.com:Liberty3000/sc2.git
cd sc2
pip install -e .
```

<img align="center" src="https://lh3.googleusercontent.com/9iFtbADMvedQJRm_IjHb2tT5swr03ZUdK2CQPeDmNocJKU9dYMySEjP_kLs9Iyx8PputYY8xbaRCcaYas7GAjgRIRQ2-xSnItquZMPc=w1440" width="1000"/>

______________
### Micro-Strategy Training Algorithms
| Algorithm                                 | Module    | Class     | Reference                                            |
|-------------------------------------------|-----------|-----------|------------------------------------------------------|
| Monte-Carlo Policy Gradient               | reinforce | REINFORCE | [arXiv:1701.07274](https://arxiv.org/abs/1701.07274) |
| Advantage Actor-Critic                    | a2c       | A2C       | [arXiv:1602.01783](https://arxiv.org/abs/1602.01783) |
| Asynchronous Advantage Actor-Critic       | a3c       | A3C       | [arXiv:1602.01783](https://arxiv.org/abs/1602.01783) |
| Deep Deterministic Policy Gradients       | ddpg      | DDPG      | [arXiv:1509.02971](https://arxiv.org/abs/1509.02971) |
| Proximal Policy Optimization              | ppo       | PPO       | [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) |
| Generative Adversarial Imitation Learning | gail      | GAIL      | [arXiv:1606.03476](https://arxiv.org/abs/1606.03476) |
| State-Action-Reward-State-Action          | sarsa     | SARSA     | [arXiv:1701.07274](https://arxiv.org/abs/1701.07274) |
| Deep Q-Network                            | dqn       | DQN       | [arXiv:1312.5602](https://arxiv.org/abs/1312.5602)   |

### Macro-Strategy Training Algorithms
| Algorithm                                 | Module | Class | Reference                                            |
|-------------------------------------------|--------|-------|------------------------------------------------------|
| Tabular SARSA                             | table  | Table | [arXiv:1701.07274](https://arxiv.org/abs/1701.07274) |
| Tabular Q-Learning                        | table  | Table | [arXiv:1701.07274](https://arxiv.org/abs/1701.07274) |
