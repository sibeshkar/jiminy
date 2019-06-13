import jiminy.gym as gym
import jiminy.wrappers.experimental
from jiminy import envs, spaces
from jiminy.wrappers import gym_core_sync
from jiminy.wrappers.blocking_reset import BlockingReset
from jiminy.wrappers.diagnostics import Diagnostics
from jiminy.wrappers.gym_core import GymCoreAction, GymCoreObservation, CropAtari
from jiminy.wrappers.joint import Joint
from jiminy.wrappers.logger import Logger
from jiminy.wrappers.monitoring import Monitor
from jiminy.wrappers.multiprocessing_env import WrappedMultiprocessingEnv, EpisodeID
from jiminy.wrappers.recording import Recording
from jiminy.wrappers.render import Render
from jiminy.wrappers.throttle import Throttle
from jiminy.wrappers.time_limit import TimeLimit
from jiminy.wrappers.timer import Timer
from jiminy.wrappers.vectorize import Vectorize, Unvectorize, WeakUnvectorize
from jiminy.wrappers.vision import Vision


def wrap(env):
    return Timer(Render(Throttle(env)))

def WrappedVNCEnv():
    return wrap(envs.VNCEnv())

def WrappedCoreVNCEnv():
    return wrap(envs.CoreVNCEnv())

def WrappedGymCoreEnv(gym_core_id, fps=None, rewarder_observation=False):
    # Don't need to store the ID on the instance; it'll be retrieved
    # directly from the spec
    env = wrap(envs.VNCEnv(fps=fps))
    if rewarder_observation:
        env = GymCoreObservation(env, gym_core_id=gym_core_id)
    return env

def WrappedGymCoreSyncEnv(gym_core_id, fps=60, rewarder_observation=False):
    spec = gym.spec(gym_core_id)
    env = gym_core_sync.GymCoreSync(BlockingReset(wrap(envs.VNCEnv(fps=fps))))
    if rewarder_observation:
        env = GymCoreObservation(env, gym_core_id=gym_core_id)
    elif spec._entry_point.startswith('gym.envs.atari:'):
        env = CropAtari(env)

    return env

def WrappedFlashgamesEnv():
    keysym = spaces.KeyEvent.by_name('`').key
    return wrap(envs.VNCEnv(probe_key=keysym))

def WrappedInternetEnv(*args, **kwargs):
    return wrap(envs.InternetEnv(*args, **kwargs))

def WrappedStarCraftEnv(*args, **kwargs):
    return wrap(envs.StarCraftEnv(*args, **kwargs))

def WrappedGTAVEnv(*args, **kwargs):
    return wrap(envs.GTAVEnv(*args, **kwargs))

def WrappedWorldOfGooEnv(*args, **kwargs):
    return wrap(envs.WorldOfGooEnv(*args, **kwargs))
