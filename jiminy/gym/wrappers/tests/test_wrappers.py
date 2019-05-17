import jiminy.gym as gym
from jiminy.gym import error
from jiminy.gym import wrappers
from jiminy.gym.wrappers import SkipWrapper

import tempfile
import shutil


def test_skip():
    every_two_frame = SkipWrapper(2)
    env = gym.make("FrozenLake-v0")
    env = every_two_frame(env)
    obs = env.reset()
    env.render()

def test_no_double_wrapping():
    temp = tempfile.mkdtemp()
    try:
        env = gym.make("FrozenLake-v0")
        env = wrappers.Monitor(env, temp)
        try:
            env = wrappers.Monitor(env, temp)
        except error.DoubleWrapperError:
            pass
        else:
            assert False, "Should not allow double wrapping"
        env.close()
    finally:
        shutil.rmtree(temp)
