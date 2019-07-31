from jiminy.vectorized.core import Env
# import time
# from jiminy.wrappers.experimental import RepresentationWrapper
from jiminy import vectorized
from jiminy.representation.structure import utils, JiminyBaseObject
from jiminy.utils.webloader import WebLoader
from jiminy.envs import SeleniumWoBEnv
from jiminy.representation.inference.betadom import BaseModel
from jiminy.utils.ml import Vocabulary
import os
import numpy as np
import queue
import threading

class betaDOMInstance(vectorized.ObservationWrapper):
    def __init__(self, screen_shape=(300, 300)):
        self.objectList = list()
        self.pixels = None
        self.flist = []
        self.screen_shape = screen_shape
        self.lock_queue = threading.Lock()
        self.frame_queue = queue.Queue()
        self.die = False

        # build model processor
        self.vocab = Vocabulary(["START", "text", "input", "checkbox", "button", "click", "END"])
        self.model = BaseModel(screen_shape=screen_shape, vocab=self.vocab)
        self.model.create_model()
        self.fp_thread = threading.Thread(target=self.frame_processor, args=())
        self.fp_thread.start()

    def frame_processor(self):
        i = 0
        while not self.die:
            frame = self.frame_queue.get()
            if i % 10 == 0:
                self.objectList = self.model.forward_pass(frame)
                print(self.objectList)

    def _observation(self, obs):
        if isinstance(obs, WebLoader):
            fname = utils.saveScreenToFile(obs.driver)
            self.flist.append(fname)
            if fname is None or fname == "":
                raise ValueError("fname for screenshot can not be null")
            self.pixels = utils.getPixelsFromFile(fname)
            self.objectList = [JiminyBaseObject(betaDOM=self, seleniumDriver=obs.driver, seleniumObject=obj)
                    for obj in obs.getRawObjectList(screen_shape=(300,300))]
            self.objectList = utils.remove_ancestors(self.objectList)
            self.query = JiminyBaseObject(betaDOM=self, seleniumDriver=obs.driver, seleniumObject=obs.getInstructionFields())
        elif isinstance(obs, np.ndarray):
            with self.lock_queue:
                self.frame_queue.put(obs)

    def __str__(self):
        jsonstring = "{{\n \"screenshot_img_path\": \"{}\",\n".format(self.flist[-1])
        jsonstring += "\"base_object_list\" : ["
        jsonstring += ",\n".join([str(obj) for obj in self.objectList])
        jsonstring += "\n]\n}"
        return jsonstring

    def close(self):
        with self.frame_queue:
            self.die = True
        self.fp_thread.join()

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
    wobenv = SeleniumWoBEnv(screen_shape=(300,300))
    jiminy_home = os.getenv("JIMINY_ROOT")
    wobenv.configure(_n=1, remotes=["file:///{}/miniwob-plusplus/html/miniwob/click-checkboxes.html".format(jiminy_home)])
    betadom = betaDOM(wobenv)
    obs = betadom.reset()
    betadom.observation(obs)

    betadom.observation([np.zeros([300, 300, 3])])

    wobenv.close()
