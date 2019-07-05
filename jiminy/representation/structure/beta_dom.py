from jiminy.vectorized.core import Env
import time
# from jiminy.wrappers.experimental import RepresentationWrapper
from jiminy import vectorized
from jiminy.representation.structure import utils, JiminyBaseObject
from jiminy.utils.webloader import WebLoader
from jiminy.envs import SeleniumWoBEnv
from jiminy.spaces import vnc_event
import os

class betaDOMInstance(vectorized.ObservationWrapper):
    def __init__(self, screen_shape=(300, 300)):
        self.objectList = list()
        self.pixels = None
        self.flist = []
        self.screen_shape = screen_shape
    def _observation(self, obs):
        if isinstance(obs, WebLoader):
            fname = utils.saveScreenToFile(obs.driver)
            self.flist.append(fname)
            if fname is None or fname == "":
                raise ValueError("fname for screenshot can not be null")
            self.pixels = utils.getPixelsFromFile(fname)
            self.objectList = [JiminyBaseObject(betaDOM=self, seleniumDriver=obs.driver, seleniumObject=obj)
                    for obj in obs.getRawObjectList(screen_shape=(300,300))]
            self.query = JiminyBaseObject(betaDOM=self, seleniumDriver=obs.driver, seleniumObject=obs.getInstructionFields())
        elif isinstance(self, np.array):
            """
            TODO: build this
            """
            raise NotImplementedError

    def __str__(self):
        jsonstring = "{{\n \"screenshot_img_path\": \"{}\",\n".format(self.flist[-1])
        jsonstring += "\"base_object_list\" : ["
        jsonstring += ",\n".join([str(obj) for obj in (self.objectList + [self.query])])
        jsonstring += "\n]\n}"
        return jsonstring

class betaDOM(vectorized.ObservationWrapper):
    """
    This object maintains the repesentation of the pixels
    seen by the environment

    This contains all object local to the said representation
    """
    def __init__(self, env=None):
        assert (not env is None), "Env passed to ClickableSpace can not be {}".format(env)
        assert isinstance(env, Env), "Env passed to ClickableSpace can not be {}, expected: jiminy.vectorized.Env".format(env)
        self.env = env
        self.betadom_instance_list = [betaDOMInstance(self.env.screen_shape) for _ in range(self.n)]

    def _observation_runner(self, i, obs):
        self.betadom_instance_list[i].observation(obs)
        return self.betadom_instance_list[i]

    def _reset_runner(self, index):
        return self.env.reset_runner(index)

    def _reset(self):
        return self.env.reset()

    def __str__(self):
        jsonstring = "{\n \"instance_list\": [\n"
        jsonstring += ",\n".join([str(instance) for instance in self.betadom_instance_list])
        jsonstring += "]\n}"
        return jsonstring

    def _close(self):
        self.env.close()

    def _step_runner(self, index, action):
        return self.env.step_runner(index, action)

if __name__ == "__main__":
    wobenv = SeleniumWoBEnv()
    jiminy_home = os.getenv("JIMINY_ROOT")
    wobenv.configure(_n=1, remotes=["file:///{}/miniwob-plusplus/html/miniwob/click-button.html".format(jiminy_home)])
    betadom = betaDOM(wobenv)
    obs = betadom.reset()
    betadom.observation(obs)
    print(betadom)
    wobenv.close()
