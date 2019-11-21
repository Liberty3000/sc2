import importlib, json, os, sys, time, threading
from pysc2 import maps
from pysc2.env import sc2_env
from sc2.loop import loop
import torch as th

def launch(players, agent, saver, worker):
    from sc2.config.asynch import flags

    id = os.path.join(saver, worker)
    os.makedirs(id, exist_ok=True)

    with open(os.path.join(saver,'config.json'),'w+') as f:
        d = flags.flags_by_module_dict()
        d = {k:{v.name:v.value for v in vs} for k,vs in d.items()}
        json.dump({'id':id, **d}, f, indent=2)

    if flags.agent_weights is not None and flags.agent_weights != '':
        print('loading weights from {}...'.format(flags.agent_weights))
        state_dict = th.load(flags.agent_weights)
        agent.model.load_state_dict(state_dict)
        print('successfully loaded weights.\n')

    if not bool(flags.train):
        if hasattr(agent,'policy') and hasattr(agent.policy,'greedy'):
            agent.policy.epsilon = agent.policy.epsilon_min
            agent.policy.greedy  = True

    agent_interface_format = sc2_env.parse_agent_interface_format(
                             feature_screen=flags.feature_screen_size,
                             feature_minimap=flags.feature_minimap,
                             action_space=flags.action_space,
                             use_feature_units=flags.use_feature_units,
                             rgb_screen=flags.rgb_screen_size)
    with sc2_env.SC2Env(players=players, map_name=flags.map,
                        agent_interface_format=agent_interface_format,
                        step_mul=flags.agent_latency,
                        game_steps_per_episode=flags.steps,
                        disable_fog=not flags.fog_of_war,
                        visualize=flags.render) as env:
        loop(agent, env, flags.episodes, train=flags.train,
        bsize=flags.bsize, saver=saver, id=worker)

def run():
    players = [{'type':flags.agent,'race':flags.agent_race,
                'agent':sc2_env.Agent(sc2_env.Race[flags.agent_race])}]

    if maps.get(flags.map).players > 1:
        diff = sc2_env.Difficulty[flags.opponent_diff]
        race = sc2_env.Race[flags.opponent_race]
        players += [{'type': flags.opponent, 'race': flags.opponent_race,
                     'agent':sc2_env.Bot(race, diff) if flags.opponent_diff
                     else sc2_env.Agent(race), 'diff': flags.opponent_diff}]

    agent_type = players[0]['type']
    id = '{}_{}'.format(agent_type, time.strftime('%a,%b.%d.%Y-%I.%M.%S.%p'))
    saver = os.path.join(flags.saver, id)
    os.makedirs(saver, exist_ok=True)

    mod,obj = agent_type.rsplit('.', 1)
    Agent = getattr(importlib.import_module(mod), obj)

    mod,obj = flags.controller.rsplit('.', 1)
    Controller = getattr(importlib.import_module(mod), obj)

    agent = Agent(device=th.device(flags.device),
    model = Controller(),
    counter = th.multiprocessing.Value('i', 0),
    lock = th.multiprocessing.Lock())

    open(os.path.join(saver,'logger.log'),'a+').close()
    open(os.path.join(saver,'training_stats.csv'),'a+').close()
    agents = [agent]

    players = [player['agent'] for player in players]

    processes = []
    for worker in range(flags.workers):
        worker_id = '{}_worker-{}'.format(id, worker)
        args = (players, agents, saver, os.path.join(saver, worker_id))
        p = th.multiprocessing.Process(target=launch, args=args)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == '__main__':
    from absl import flags
    flags = flags.FLAGS
    flags(sys.argv)
    from sc2.config.asynch import flags
    run()
