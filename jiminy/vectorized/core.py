from jiminy.gym import Env, Wrapper, ObservationWrapper, RewardWrapper, ActionWrapper
from jiminy.gym import spaces
from jiminy import error

class Env(Env):
    """Base class capable of handling vectorized environments.
    """
    metadata = {
        # This key indicates whether an env is vectorized (or, in the case of
        # Wrappers where autovectorize=True, whether they should automatically
        # be wrapped by a Vectorize wrapper.)
        'runtime.vectorized': True,
    }

    # Number of remotes. User should set this.
    n = None
    def reset_runner(self, index):
        return self._reset_runner(index)

    def _reset_runner(self, index):
        raise NotImplementedError

    def step_runner(self, index, action):
        return self._step_runner(index, action)

    def _step_runner(self, index, action):
        raise NotImplementedError

class Wrapper(Env, Wrapper):
    """Use this instead of gym.Wrapper iff you're wrapping a vectorized env,
    (or a vanilla env you wish to be vectorized).
    """
    # If True and this is instantiated with a non-vectorized environment,
    # automatically wrap it with the Vectorize wrapper.
    autovectorize = True

    def __init__(self, env):
        super(Wrapper, self).__init__(env)
        if not env.metadata.get('runtime.vectorized'):
            if self.autovectorize:
                # Circular dependency :(
                from jiminy import wrappers
                env = wrappers.Vectorize(env)
            else:
                raise error.Error('This wrapper can only wrap vectorized envs (i.e. where env.metadata["runtime.vectorized"] = True), not {}. Set "self.autovectorize = True" to automatically add a Vectorize wrapper.'.format(env))

        self.env = env

    @property
    def n(self):
        return self.env.n

    def configure(self, **kwargs):
        self.env.configure(**kwargs)


class ObservationWrapper(Wrapper, ObservationWrapper):
    def _observation(self, observation):
        assert (isinstance(observation, list) and len(observation) == self.n), "Expected observation to be list of size {}".format(self.n, observation)
        obs_list = []
        for i, obs in enumerate(observation):
            obs_list.append(self.observation_runner(i, obs))
        return obs_list

    def _step_runner(self, index, action):
        observation, reward, done, info = self.env.step_runner(index, action)
        return self.observation_runner(index, observation), reward, done, info

    def observation_runner(self, index, observation):
        return self._observation_runner(index, observation)

    def _observation_runner(self, index, observation):
        raise NotImplementedError

class RewardWrapper(Wrapper, RewardWrapper):
    pass

class ActionWrapper(Wrapper, ActionWrapper):
    def _step_runner(self, index, action):
        action = self.action_runner(action)
        return self.env.step_runner(index, action)

    def action_runner(self, action):
        return self._action_runner(action)
    def _action_runner(self, action):
        raise NotImplementedError
    def reverse_action_runner(self, action):
        return self._reverse_action_runner(action)
    def _reverse_action_runner(self, action):
        raise NotImplementedError
    pass
