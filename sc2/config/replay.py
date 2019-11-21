from absl import flags
from sc2 import load_path, make_path

flags.DEFINE_string (             'loader', default=load_path(), help='')
flags.DEFINE_string (              'saver', default=make_path('experiments','extract'), help='')
flags.DEFINE_integer(    'observed_player', default=1, help='')
flags.DEFINE_integer(                'fps', default=30, help='')
flags.DEFINE_integer(           'episodes', default=2**10, help='')
flags.DEFINE_bool   (         'fog_of_war', default=True, help='')
flags.DEFINE_integer(      'agent_latency', default=8, help='')
flags.DEFINE_integer('feature_screen_size', default=84,help='')
flags.DEFINE_integer(    'feature_minimap', default=64,help='')
flags.DEFINE_bool   (  'use_feature_units', default=True, help='')
flags.DEFINE_integer(    'rgb_screen_size', default=None, help='')
flags.DEFINE_string (       'action_space', default=None, help='')
flags.DEFINE_bool   (        'save_replay', default=False, help='')
flags.DEFINE_bool   (             'render', default=False,help='')

flags = flags.FLAGS
