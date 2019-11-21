import glob, importlib, json, mpyq, os, six, sys, time, traceback
import numpy as np
from absl import flags
from pysc2 import maps, run_configs
from pysc2.env import sc2_env
from pysc2.lib import features, protocol, remote_controller
from pysc2.env import available_actions_printer
from s2clientprotocol import sc2api_pb2 as sc_pb
from sc2.util import preprocess_observation

# screen features
screen_features = ['height_map', 'visibility_map', 'creep', 'power', 'player_id', 'player_relative',
                   'unit_type', 'selected', 'unit_hit_point', 'unit_hit_point_ratio', 'unit_energy',
                   'unit_energy_ratio', 'unit_shield', 'unit_shield_ratio', 'unit_density',
                   'unit_density_ratio', 'effects']
# global features
minimap_features = ['height_map', 'visibility_map', 'creep', 'camera', 'player_id', 'player_relative',
                    'selected']
# non-spatial features
other_features = ['player', 'game_loop', 'score_cumulative', 'available_actions', 'single_select',
                  'multi_select', 'cargo', 'cargo_slots_available', 'build_queue', 'control_groups']

def get_game_version(replay_data):
    replay_io = six.BytesIO()
    replay_io.write(replay_data)
    replay_io.seek(0)
    archive = mpyq.MPQArchive(replay_io).extract()
    metadata = json.loads(archive[b'replay.gamemetadata.json'].decode('utf-8'))
    version = metadata['GameVersion']
    return '.'.join(version.split('.')[:-1])

def parse(replay):
    run_config = run_configs.get()

    interface = sc_pb.InterfaceOptions()
    interface.score = True
    interface.feature_layer.resolution.x = flags.feature_screen_size
    interface.feature_layer.resolution.y = flags.feature_screen_size
    interface.feature_layer.minimap_resolution.x = flags.feature_minimap
    interface.feature_layer.minimap_resolution.y = flags.feature_minimap

    replay_data = run_config.replay_data(replay)
    start_replay = sc_pb.RequestStartReplay(
                   replay_data=replay_data,
                   options=interface,
                   disable_fog=not flags.fog_of_war,
                   observed_player_id=flags.observed_player)
    game_version = get_game_version(replay_data)
    print('StarCraft II', game_version, replay)

    with run_config.start(game_version=game_version) as controller:
        info = controller.replay_info(replay_data)

        print('REPLAY'.center(60,'_'))
        print(info)
        print('_' * 60)

        map_path = info.local_map_path
        print(map_path)

        if map_path: start_replay.map_data = run_config.map_data(map_path)

        nplayers = maps.get(map_path.split('/')[-1].split('.')[0]).players

        for observed_player in range(1, nplayers+1):
            controller.start_replay(start_replay)

            screen_features = []
            global_features = []
            nonspt_features = []
            actions = []
            action_args = []
            rewards = []
            terminals = []

            try:
                feat = features.features_from_game_info(controller.game_info(), use_feature_units=flags.use_feature_units)

                temp = replay.split('\\')[-1].split('.')[-2].split('/')[-1]
                save_as = temp + '_player_{}'.format(observed_player)
                datafold = os.path.join(saver, save_as)

                itr = 0
                while True:
                    itr += 1
                    beg = time.time()
                    controller.step(flags.agent_latency)
                    obs = controller.observe()
                    obs_t = feat.transform_obs(obs)

                    screen_features.append(obs_t['feature_screen'])
                    global_features.append(obs_t['feature_minimap'])

                    for action in obs.actions:
                        break
                        try:
                            func = feat.reverse_action(action).function
                            args = feat.reverse_action(action).arguments
                            print(func, args)

                            actions.append(func)
                            action_args.append(args)

                            with open(os.path.join(datafold, 'action.txt'), 'a+', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerows([obs_t['game_loop'].tolist(), [func], [args]])
                        except ValueError:
                            pass

                    if obs.player_result: break

                # screen features
                screen_feats = np.stack(np.asarray(screen_features), axis=0)
                np.save(os.path.join(saver,'screen-{}.npy'.format(save_as)), screen_feats)
                # minimap features
                global_feats = np.stack(np.asarray(global_features), axis=0)
                np.save(os.path.join(saver,'global-{}.npy'.format(save_as)), global_feats)

            except:
                print(traceback.format_exc())
            print(' score:', obs.observation.score.score)
            print('result:', obs.player_result)
            print('\n\n')


if __name__ == '__main__':
    flags = flags.FLAGS
    flags(sys.argv)
    from sc2.config.replay import flags

    global saver
    saver = os.path.join(flags.saver, 'extract_{}'.format(time.strftime('%b.%d.%Y-%I.%M.%S.%p')))
    os.makedirs(saver, exist_ok=True)

    for replay in glob.glob(flags.loader):
        print(replay)
        parse(replay)
