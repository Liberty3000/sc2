import importlib, json, os, sys, time, torch as th
from pysc2 import maps
from pysc2.env import sc2_env
from sc2.loop import loop
from sc2.config.simulate import flags

def launch(players):
    agents= [player['agent'] for player in players]
    agent_interface_format = sc2_env.parse_agent_interface_format(
                             feature_screen=flags.feature_screen_size,
                             feature_minimap=flags.feature_minimap,
                             action_space=flags.action_space,
                             use_feature_units=flags.use_feature_units,
                             rgb_screen=flags.rgb_screen_size)
    with sc2_env.SC2Env(players=agents,
                        map_name=flags.map,
                        agent_interface_format=agent_interface_format,
                        step_mul=flags.agent_latency,
                        game_steps_per_episode=flags.steps,
                        disable_fog=not flags.fog_of_war,
                        visualize=flags.render) as env:
        id = '{}_{}'.format(players[0]['type'], time.strftime('%a,%b.%d.%Y-%I.%M.%S.%p'))
        print(id,'\n')
        saver = os.path.join(flags.saver, id)
        os.makedirs(saver, exist_ok=True)
        print(saver)

        with open(os.path.join(saver,'config.json'),'w+') as f:
            d = flags.flags_by_module_dict()
            d = {k:{v.name:v.value for v in vs} for k,vs in d.items()}
            json.dump({'id':id,**d}, f, indent=2)

        mod,obj = flags.controller.rsplit('.', 1)
        Controller = getattr(importlib.import_module(mod), obj)
        model = Controller()

        mod,obj = players[0]['type'].rsplit('.', 1)
        Agent = getattr(importlib.import_module(mod), obj)
        agent = Agent(model=model, device=th.device(flags.device))
        agents[0] = agent

        if flags.agent_weights is not None and flags.agent_weights != '':
            print('loading pre-trained weights from {}...'.format(flags.agent_weights))
            state_dict = th.load(flags.agent_weights)
            agent.model.load_state_dict(state_dict)
            print('successfully loaded weights.\n')

        train = False if not hasattr(agent, 'update') else flags.train
        if not bool(train):
            if hasattr(agent,'policy') and hasattr(agent.policy,'greedy'):
                agent.policy.epsilon = agent.policy.epsilon_min
                agent.policy.greedy = True

        loop(agents, env, episodes=flags.episodes, train=train,
             bsize=flags.bsize, saver=flags.saver, id=id)

def run():
    players = [{'type':flags.agent,'race':flags.agent_race,
                'agent':sc2_env.Agent(sc2_env.Race[flags.agent_race])}]

    if maps.get(flags.map).players > 1:
        diff = sc2_env.Difficulty[flags.opponent_diff]
        race = sc2_env.Race[flags.opponent_race]
        players += [{'type': flags.opponent, 'race': flags.opponent_race,
                     'agent':sc2_env.Bot(race, diff) if flags.opponent_diff
                     else sc2_env.Agent(race), 'diff': flags.opponent_diff}]

    launch(players)

if __name__ == '__main__':
    flags(sys.argv)
    from sc2.config.simulate import flags
    run()
