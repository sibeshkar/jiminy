from jiminy.gym.envs.mujoco.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from jiminy.gym.envs.mujoco.ant import AntEnv
from jiminy.gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from jiminy.gym.envs.mujoco.hopper import HopperEnv
from jiminy.gym.envs.mujoco.walker2d import Walker2dEnv
from jiminy.gym.envs.mujoco.humanoid import HumanoidEnv
from jiminy.gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from jiminy.gym.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv
from jiminy.gym.envs.mujoco.reacher import ReacherEnv
from jiminy.gym.envs.mujoco.swimmer import SwimmerEnv
from jiminy.gym.envs.mujoco.humanoidstandup import HumanoidStandupEnv
from jiminy.gym.envs.mujoco.pusher import PusherEnv
from jiminy.gym.envs.mujoco.thrower import ThrowerEnv
from jiminy.gym.envs.mujoco.striker import StrikerEnv
