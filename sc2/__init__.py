import os, sys

def make_path(*args):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(BASE_DIR, *args)

def loader_path():
    paths = dict(
    darwin=r'/Applications/StarCraft II/Replays/*/*.SC2Replay',
    linux=r'/home/StarCraftII/Replays/*/*.SC2Replay',
    windows=r'C:\Program Files (x86)\StarCraft II\Replays\*\*.SC2Replay')
    return paths.get(sys.platform)
