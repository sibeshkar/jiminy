from jiminy.wrappers.experimental import RepresentationWrapper
from jiminy.vectorized import Env
from jiminy.gym import spaces
import numpy as np
from selenium.webdriver.remote.webelement import WebElement

class ClickableSpace(RepresentationWrapper):
    def __init__(self, clickTarget, env=None):
        assert (not env is None), "Env passed to ClickableSpace can not be {}".format(env)
        assert isinstance(env, Env), "Env passed to ClickableSpace can not be {}, expected: jiminy.vectorized.Env".format(env)
        super(ClickableSpace, self).__init__(env)
        """
        Observations that are possible are:
            1. Left-down
            2. Right-down
            3. Hover
            4. OOB -- out of bounds
        """
        self.observation_space = spaces.discrete(4)
        self.action_space = spaces.ClickActionSpace

    def _observation(self, obs):
        if isinstance(obs, np.array):
            # obs is pixels
            return self._observation_from_pixels(obs)
        else if isinstance(obs, WebElement):
            # obs is selenium object
            return self._observation_from_web_element(obs)

    def _observation_from_web_element(self, obs):
        """
        TODO: implement this
        """
        return self.observation_space.sample()

    def _observation_from_pixels(self, pixels):
        """
        TODO: implement this
        """
        return self.observation_space.sample()

if __name__ == "__main__":
    cs = ClickableSpace(None, env=Env())
