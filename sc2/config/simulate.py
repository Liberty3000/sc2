from absl import flags
from pysc2.env import sc2_env
from sc2 import make_path

flags.DEFINE_string (              'agent', default='sc2.agents.dqn.DQN', help='')
flags.DEFINE_string (         'controller', default='sc2.controllers.fully_conv.FullyConv', help='')

flags.DEFINE_bool   (              'train', default=True, help='')
flags.DEFINE_integer(              'bsize', default=64, help='')

flags.DEFINE_string (                'map', default='DefeatRoaches', help='')
flags.DEFINE_bool   (         'fog_of_war', default=True,help='')
flags.DEFINE_integer(           'episodes', default=2**19,help='')
flags.DEFINE_integer(              'steps', default=2**25, help='')

flags.DEFINE_string (      'agent_weights', default='', help='')
flags.DEFINE_string (   'opponent_weights', default='', help='')

flags.DEFINE_integer(      'agent_latency', default=10, help='')
flags.DEFINE_enum   (         'agent_race', default='random', enum_values=sc2_env.Race._member_names_, help='')

flags.DEFINE_string (           'opponent', default='bot', help='')
flags.DEFINE_enum   (      'opponent_race', default='random', enum_values=sc2_env.Race._member_names_, help='')
flags.DEFINE_string (      'opponent_diff', default='very_easy', help='')

flags.DEFINE_integer('feature_screen_size', default=84,   help='')
flags.DEFINE_integer(    'feature_minimap', default=64,   help='')
flags.DEFINE_bool   (  'use_feature_units', default=True, help='')
flags.DEFINE_integer(    'rgb_screen_size', default=None,  help='')
flags.DEFINE_string (       'action_space', default=None,  help='')

flags.DEFINE_string (              'saver', default=make_path('experiments'), help='')
flags.DEFINE_bool   (            'verbose', default=True,  help='')
flags.DEFINE_bool   (        'save_replay', default=True,  help='')
flags.DEFINE_bool   (             'render', default=False, help='')
flags.DEFINE_string (             'device', default='cpu', help='')

flags = flags.FLAGS
