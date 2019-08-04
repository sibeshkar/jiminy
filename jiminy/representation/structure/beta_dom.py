import jiminy
import sys
from jiminy.vectorized.core import Env
# import time
# from jiminy.wrappers.experimental import RepresentationWrapper
from jiminy import vectorized
from jiminy.representation.structure import utils, JiminyBaseObject, JiminyBaseInstancePb2
from jiminy.utils.webloader import WebLoader
from jiminy import gym
from jiminy.representation.inference.betadom import BaseModel
from jiminy.utils.ml import Vocabulary
# import os
import time
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
        super(betaDOM, self).__init__(env)

    def _observation_runner(self, i, obs):
        return self.betadom_instance_list[i].observation(obs)

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

    def configure(self, *args, **kwargs):
        if "screen_shape" in kwargs:
            self.screen_shape = kwargs["screen_shape"]
            del kwargs["screen_shape"]
        self.env.configure(*args, **kwargs)
        self.betadom_instance_list = [JiminyBaseInstancePb2() for _ in
                range(self.n)]

# testModeName = "selenium"
testModeName = "VNC.Core-v0"

if __name__ == "__main__":
    env = gym.make(testModeName)
    env = jiminy.actions.experimental.SoftmaxClickMouse(env)
    env = betaDOM(env)
    env.configure(screen_shape=(300,300), env='sibeshkar/wob-v1', task='ClickButton', remotes='vnc://0.0.0.0:5901+15901')
    obs = env.reset()
    first_set = False
    while True:
        a = env.action_space.sample()
        obs, reward, is_done, info = env.step([a])
        if obs[0] is None:
            if not first_set:
                print("Env is still resetting...", end="")
                first_set = True
            else:
                print(".", end="")
            sys.stdout.flush()
            continue
        print()
        break

    for _ in range(5000):
        time.sleep(0.05)
        a = env.action_space.sample()
        obs, reward, is_done, info = env.step([a])
        # obs = env.observation(obs)
        if obs[0] is None:
            print("Env is resetting...")
            continue
        print("Sampled action: ", a)
        if not obs is None:
            print("Observation", str(obs))
        print("Reward", reward)
        print("Is done", is_done)
        print("Info", info)
        env.render()
    env.close()
