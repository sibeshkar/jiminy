from jiminy.vectorized.core import Env
from jiminy.wrappers.experimental import RepresentationWrapper
import os

class betaDOM(RepresentationWrapper):
    """
    This object maintains the repesentation of the pixels
    seen by the environment

    This contains all object local to the said representation
    """
    def __init__(self, env=None, actionableStateList=None):
        if env is None:
            raise ValueError("betaDOM needs Env")
        self.env = env
        self.objectList = list()
        self.pixels = None
        if not actionableStateList is None:
            self.actionableStateList = actionableStateList
        else:
            # if no actionableStateList is provided: use the default actions
            # provided by the environment
            self.actionableStateList = Env.action_space

def betaDOMFromSeleniumWebDriver(seleniumWebDriver):
    betadom = betaDOM(env=Env())
    fname = utils.saveScreenToFile(seleniumWebDriver)
    if fname is None or fname == "":
        raise ValueError("fname for a screenshot can not be null")
    betadom.pixels = getPixelsFromFile(fname)

if __name__ == "__main__":
    betadom = betaDOM(env=Env())
    betadom._step(None)
