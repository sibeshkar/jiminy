from jiminy.vectorized.core import Env
# from jiminy.wrappers.experimental import RepresentationWrapper
from jiminy import vectorized
from jiminy.representation.structure import utils, JiminyBaseObject
from jiminy.utils.webloader import WebLoader
import os

class betaDOMInstance(vectorized.ObservationWrapper):
    def __init__(self):
        self.objectList = list()
        self.pixels = None
        self.flist = []
    def _observation(self, obs):
        if isinstance(obs, WebLoader):
            fname = utils.saveScreenToFile(obs.driver)
            self.fname.append(fname)
            if fname is None or fname == "":
                raise ValueError("fname for screenshot can not be null")
            self.pixels = utils.getPixelsFromFile(fname)
            self.objectList = [JiminyBaseObject(betaDOM=self, seleniumDriver=obs.driver, seleniumObject=obj)
                    for obj in obs.getRawObjectList()]
        elif isinstance(self, np.array):
            """
            TODO: build this
            """
            raise NotImplementedError

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
        self.betadom_instance_list = [betaDOMInstance() for _ in range(self.n)]

    def _observation(self, obs):
        assert (isinstance(obs, list) and len(obs) == self.n), "Expected observation to be list of size {}".format(self.n, obs)
        for i, ob in enumerate(obs):
            self.betadom_instance_list[i].observation(ob)

    def _reset(self):
        self.env.reset()
        self.

if __name__ == "__main__":
    betadom = betaDOM(env=Env())
    webloader = WebLoader("Firefox")
    betadom.observation(webloader)
