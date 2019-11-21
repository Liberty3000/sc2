import os, numpy as np, torch as th
from pysc2.lib import actions, features

def compute_returns(vprime, rewards, masks, gamma=0.99):
    R,returns = vprime.detach(),[]
    for t in reversed(range(len(rewards))):
        R = rewards[t] + gamma * R * masks[t]
        returns.insert(0, R)
    return returns

def normalize_rewards(rewards, gamma=0.99):
    rewards = torch.tensor(discount_rewards(rewards, gamma))
    denom = rewards.std() + np.finfo(np.float32).eps.item()
    rewards = (rewards - rewards.mean())/(denom)
    return rewards

def estimate_advantage(vprime, values, rewards, masks, gamma=0.99, tau=0.95):
    values = th.cat((values,vprime),dim=0).detach()
    gae,returns = 0,[]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * masks[t] - values[t]
        gae = delta + gamma * tau * masks[t] * gae
        returns.insert(0, gae + values[t])
    return returns

def mask_actions(legal_actions, action_space=549):
    mask = th.zeros((1,action_space))
    mask[:,legal_actions] = 1
    return mask

def parameterize_action(action_id, args, available_actions):
    action = available_actions[action_id]
    output = []
    for arg in actions.FUNCTIONS[action].args:
        if arg.name in ['screen','screen2','minimap']:
            output.append([args[-1], args[-2]])
        else:
            output.append([0])
    return actions.FunctionCall(action, output)

def preprocess_observation(obs, verbose=False):
    global_features = np.asarray(obs.observation.feature_minimap)
    global_features = th.from_numpy(global_features).float().unsqueeze(0)

    local_features = np.asarray(obs.observation.feature_screen)
    local_features= th.from_numpy(local_features).float().unsqueeze(0)

    game_loop = np.asarray(obs.observation['game_loop'])
    game_data = np.asarray(obs.observation['player'])
    control_groups = np.asarray(obs.observation['control_groups'])
    cummulative_reward = np.asarray(obs.observation['score_cumulative'])

    game_loop = th.from_numpy(game_loop).float().view(-1)
    game_data = th.from_numpy(game_data).float().view(-1)
    control_groups = th.from_numpy(control_groups).float().view(-1)
    cummulative_reward = th.from_numpy(cummulative_reward).float().view(-1)

    state_vec = th.cat((game_loop, game_data, control_groups, cummulative_reward), dim=-1)
    non_spatial_features = state_vec.unsqueeze(0)

    return global_features, local_features, non_spatial_features

# determines the spawn location of the player
# 1 if top left, 0 if bottom right
def determine_spawn(obs):
    _PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
    _PLAYER_SELF = 1

    minimap = obs.observation.feature_minimap

    player_ys, player_xs = (minimap[_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()

    return player_ys.mean() <= minimap.shape[-1] - 1

# transforms locations to be relative to our base
def relative_location(top_left, x, y, xdist, ydist):
    if not top_left:
        return [x - xdist, y - ydist]
    else:
        return [x + xdist, y + ydist]

def units_by_type(obs, unit_type):
    return [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]

def selected(obs, unit_type):
    if (len(obs.observation.single_select) > 0 and obs.observation.single_select[0].unit_type == unit_type): return True
    if (len(obs.observation.multi_select)  > 0 and obs.observation.multi_select[0].unit_type == unit_type): return True
    return False

def sparsify_reward(reward, wins, losses, draws):
    if reward < 0: losses += 1
    elif reward > 0: wins += 1
    else: draws += 1
    return wins, losses, draws

def binary_threat_map(obs, top_left=True):
    _PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
    _PLAYER_SELF = 1
    _PLAYER_HOSTILE = 4

    threat_map = []

    FRIENDLY= np.zeros(4)
    HOSTILE = np.zeros(4)

    enemy_y, enemy_x = (obs.observation.feature_minimap[_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
    for i in range(0, len(enemy_y)):
        y = int(np.ceil((enemy_y[i] + 1) / 32))
        x = int(np.ceil((enemy_x[i] + 1) / 32))
        HOSTILE[((y - 1) * 2) + (x - 1)] = 1

    friendly_y, friendly_x = (obs.observation.feature_minimap[_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
    for i in range(0, len(friendly_y)):
        y = int(np.ceil((friendly_y[i] + 1) / 32))
        x = int(np.ceil((friendly_x[i] + 1) / 32))
        FRIENDLY[((y - 1) * 2) + (x - 1)] = 1

    if not  top_left:
        HOSTILE  = HOSTILE [::-1]
        FRIENDLY = FRIENDLY[::-1]

    for i in range(0, 4):
        threat_map += [FRIENDLY[i]]
        threat_map += [HOSTILE[i]]

    return np.asarray(threat_map)

def export(agent, id):
    if hasattr(agent,'model'):
        name = str(agent.model.__class__.__name__)
        save_as = os.path.join(id, '{}.th'.format(name))
        th.save(agent.model.state_dict(), save_as)

    if hasattr(agent,'actor'):
        name = str(agent.actor.__class__.__name__)
        save_as = os.path.join(id, '{}_actor.th'.format(name))
        th.save(agent.actor.state_dict(), save_as)

    if hasattr(agent,'critic'):
        name = str(agent.critic.__class__.__name__)
        save_as = os.path.join(id, '{}_critic.th'.format(name))
        th.save(agent.critic.state_dict(), save_as)

    if hasattr(agent,'discriminator'):
        name = str(agent.discriminator.__class__.__name__)
        save_as = os.path.join(id, '{}_discriminator.th'.format(name))
        th.save(agent.discriminator.state_dict(), save_as)

def visualize_policy(spatial_logits):
    if isinstance(spatial_logits, tuple):
        self.ax.scatter(*[s.detach().numpy() for s in spatial_logits], color='red')
    else:
        self.ax.imshow(spatial_logits.squeeze().cpu().detach().numpy())
        self.ax.scatter(*action_args, color='blue')
    mp.show(block=False)
    mp.pause(1e-4)
    self.fig.canvas.draw_idle()
    try: self.fig.canvas.flush_events()
    except: pass
