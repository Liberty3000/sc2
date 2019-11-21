import logging as log, os, time, traceback, tqdm, numpy as np, pandas as pd, sys
from sc2.util import sparsify_reward, export
from pysc2.env.available_actions_printer import AvailableActionsPrinter

def loop(agents, env, episodes=1, train=False, bsize=1, time_limit=2**30, saver=None, id=None):
    worker = os.path.join(saver, id)
    fname = os.path.join(worker, 'logger.log')
    log.basicConfig(filename=fname, level=log.DEBUG)

    wins,losses,draws = 0,0,0
    record = pd.DataFrame()
    agent, ebar, steps, updates = agents[0],tqdm.tqdm(range(1,1+episodes)),0,0
    try:
        for episode in ebar:
            itr, rewards = 0, []
            timesteps = env.reset()
            obs = timesteps[0]
            while itr < time_limit:
                eban = 'ep: {}, step: {:4d}, r: {:2f}'

                if len(agents) > 1:
                    actions = [agent_.step(timestep) for (agent_,timestep) in zip(agents, timesteps)]
                else:
                    actions = [agent.step(obs)]

                action = actions[0]
                obs = env.step(actions)[0]
                reward, terminal = obs.reward, obs.last()
                rewards.append(reward)
                steps = itr = itr + 1

                if hasattr(agent, 'memory'): agent.memorize(obs)

                if hasattr(agent, 'policy') and hasattr(agent.policy, 'epsilon'):
                    # if the agent's policy has an exploration rate, display it
                    exploration_rate = 'ϵ: {:.4f}'.format(agent.policy.epsilon)
                    eban = '{}, {}'.format(exploration_rate, eban)

                ebar.set_description(eban.format(episode, steps, np.sum(rewards), wins, losses, draws))
                if terminal:
                    if len(agents) > 1:
                        # if it's a multi-player game, only a sparse ternary reward is available
                        wins, losses, draws = sparsify_reward(reward, wins, losses, draws)
                        record = record.append({'Win':wins,'Loss':losses,'Draw':draws}, ignore_index=True)
                        record.to_csv(os.path.join(worker,'performance.csv'))
                    else:
                        # if it's a single-player game, the agent may have had access to shaped intermediate rewards
                        record = record.append({'Cummulative':np.sum(rewards),'Mean Reward':np.mean(rewards),
                                                 'Max Reward':np.max(rewards)}, ignore_index=True)
                        record.to_csv(os.path.join(worker,'performance.csv'))
                    break

            if train:
                loss = agent.update(obs, bsize=bsize, worker=worker)
                updates += 1

            export(agent, worker)
            print('episode {}, {} steps. {} updates.\n'.format(episode, itr, updates))

            if saver is not None and id is not None:
                save_as = os.path.join(worker,'replays')
                os.makedirs(save_as, exist_ok=True)
                env.save_replay(save_as)

    except:
        print(traceback.format_exc())
        log.info(traceback.format_exc())
