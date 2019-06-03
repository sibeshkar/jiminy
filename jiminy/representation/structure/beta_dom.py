from jiminy.vectorized.core import Env
from jiminy.wrappers.experimental import RepresentationWrapper
from jiminy.representation.structure import utils
from jiminy.representation import WebLoader
# from jiminy.gc
import os

class betaDOM(RepresentationWrapper):
    """
    This object maintains the repesentation of the pixels
    seen by the environment

    This contains all object local to the said representation
    """
    def __init__(self, env=None, actionableStateList=None):
        assert (not env is None), "Env passed to ClickableSpace can not be {}".format(env)
        assert isinstance(env, Env), "Env passed to ClickableSpace can not be {}, expected: jiminy.vectorized.Env".format(env)
        self.env = env
        self.objectList = list()
        self.pixels = None
        if not actionableStateList is None:
            self.actionableStateList = actionableStateList
        else:
            # if no actionableStateList is provided: use the default actions
            # provided by the environment
            self.actionableStateList = Env.action_space
            if self.actionableStateList is None:
                self.actionableStateList = ["empty"]

    def getActionableStateList(self):
        return self.actionableStateList

    def _observation(self, obs):
        if isinstance(obs, WebLoader):
            self.fname = utils.saveScreenToFile(obs.driver)
            if fname is None or fname == "":
                raise ValueError("fname for screenshot can not be null")
            betadom.pixels = utils.getPixelsFromFile(fname)
            self.objectList = [JiminyBaseObject(obj) for obj in obs.getRawObjectList()]
            self.action_space = utils.flatten([obj.getActions() for obj in self.objectList])
            return self

        elif isinstance(self, np.array):
            """
            TODO: build this
            """
            raise NotImplementedError

if __name__ == "__main__":
    betadom = betaDOM(env=Env())
    webloader = WebLoader("Firefox")
    betadom.observation(webloader)
