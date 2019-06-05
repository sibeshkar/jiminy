from jiminy.envs import DummyVNCEnv
from jiminy.utils.webloader import WebLoader
from jiminy.spaces import vnc_event
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from jiminy import spaces
from jiminy.representation.structure import betaDOM
import numpy as np


class SeleniumWoBEnv(DummyVNCEnv):
    """
    Selenium based Env to interact with web pages
    """
    def __init__(self):
        self._started = False
        self.observation_space = spaces.VNCObservationSpace()
        self.action_space = spaces.VNCActionSpace()
        self.buttonmask = 0

    def configure(self, remotes=None,
                   client_id=None,
                   start_timeout=None, docker_image=None,
                   ignore_clock_skew=False, disable_action_probes=False,
                   vnc_driver=None, vnc_kwargs={},
                   replace_on_crash=False, allocate_sync=True,
                   observer=False,
                   _n=3
        ):
        assert _n == len(remotes), "Expected {} remotes but recived {}: {}".format(self.n, len(remotes), remotes)
        self.n = _n
        self.remotes = remotes
        self.web_driver_list = [WebLoader("Firefox") for _ in range(self.n)]
        for i in range(self.n):
            self.web_driver_list[i].loadPage(self.remotes[i])
        self.started = True

    def _reset(self):
        for i in range(self.n):
            self.web_driver_list[i].loadPage(self.remotes[i])
        return self.web_driver_list

    def _step(self, action_n):
        assert self.n == len(action_n), "Expected {} actions but received {}: {}".format(self.n, len(action_n), action_n)
        assert self.action_space.contains(action_n), "Expected VNCActions by received {}".format(action_n)

        reward_list = [0. for _ in range(self.n)]
        done_list = [False for _ in range(self.n)]

        for i in range(self.n):
            self._action_impl(self.web_driver_list[i], action_n[i])

        for i in range(self.n):
            reward_list[i] = self._reward(self.web_driver_list[i])
            done_list = self._get_env_data(self.web_driver_list[i])['done']

        return self.web_driver_list, reward_list, done_list, {}

    def _reward(self, web_driver):
        json_reward = self._get_env_data(web_driver)
        return float(json_reward['env_reward'])

    def _get_env_data(self, web_driver):
        try:
            return web_driver.driver.execute_script(
                "return {"
                "    'env_reward' : WOB_REWARD_GLOBAL,"
                "    'done' : WOB_DONE_GLOBAL"
                "};")
        except:
            return {
                    "env_reward" : 0.,
                    "done" :  True
                }

    def _action_impl(self, web_driver, action):
        webaction = ActionChains(web_driver.driver)
        if isinstance(action, vnc_event.KeyEvent):
            key_action = vnc_event.KeyEvent._key_sym_to_name[action]
            if key_action in Keys.__dict__:
                key_action = Keys.__dict__[key_action]
            if action.down:
                webaction.key_down(key_action)
            else:
                webaction.key_up(key_action)
        if isinstance(action, vnc_event.PointerEvent):
            body = web_driver.driver.find_element_by_tag_name('body')
            webaction.move_to_element_with_offset(body, 0, 0)
            webaction.move_by_offset(action.x, action.y)
            # TODO(prannayk) : implement other clicks except right click
            if (action.buttonmask & ~(self.buttonmask & 1)) :
                webaction.click_and_hold()
                self.buttonmask = action.buttonmask
            else:
                webaction.release()
                self.last_mouse_action = 0
        webaction.perform()

    def _close(self):
        [webdriver.driver.close() for webdriver in self.web_driver_list]

if __name__ == "__main__":
    wobenv = SeleniumWoBEnv()
    wobenv.configure(_n=2, remotes=["https://www.google.com/", "https://www.google.com/"])
    element = wobenv.web_driver_list[0].driver.find_element_by_class_name("RNNXgb")
    rect = element.rect
    x = rect['x'] + (rect['width'] / 2.)
    y = rect['y'] + (rect['height'] / 2.)
    vnc_click = vnc_event.PointerEvent(x, y, 1)
    wobenv.step([vnc_click, vnc_click])
    vnc_click = vnc_event.PointerEvent(x, y, 0)
    obs, _, _, _ = wobenv.step([vnc_click, vnc_click])
    betadom = betaDOM(wobenv)
    betadom.observation(obs)
    wobenv.close()
