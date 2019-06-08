from jiminy.envs import DummyVNCEnv
from jiminy.utils.webloader import WebLoader
from jiminy.spaces import vnc_event
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.events import EventFiringWebDriver, AbstractEventListener
from jiminy import spaces
import numpy as np
import time

class FirefoxListener(AbstractEventListener):
    def before_click(self, element, driver):
        pass

    def after_click(self, element, driver):
        pass


class SeleniumWoBEnv(DummyVNCEnv):
    """
    Selenium based Env to interact with web pages
    """
    def __init__(self):
        self._started = False
        self.observation_space = spaces.VNCObservationSpace()
        self.action_space = spaces.VNCActionSpace(buttonmasks=[0,1], screen_shape=(160, 210), event_type=1)
        self.buttonmask = 0

    def start_listener(self):
        """
        TODO(prannayk): complete this
        """
        assert (not self.web_driver_list is None), "Call configure on the environment before calling start_listener"
        raise NotImplementedError

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
        action_list = []
        for i in range(self.n):
            action_list.append(self.action_space.sample())
            action_list[-1].buttonmask = 1
        self._step(action_list)
        action_list = []
        for i in range(self.n):
            action_list.append(self.action_space.sample())
            action_list[-1].buttonmask = 0
        obs, _, _, _ = self._step(action_list)
        return obs

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
            key_action = vnc_event.KeyEvent._keysym_to_name[action]
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
            if (action.buttonmask != self.buttonmask) and action.buttonmask == 1:
                webaction.click_and_hold()
                self.buttonmask = action.buttonmask
            elif (action.buttonmask != self.buttonmask) and action.buttonmask == 0:
                webaction.release()
                self.buttonmask = action.buttonmask
        webaction.perform()

    def _close(self):
        [webdriver.driver.close() for webdriver in self.web_driver_list]

if __name__ == "__main__":
    wobenv = SeleniumWoBEnv()
    wobenv.configure(_n=1, remotes=["file:///Users/prannayk/ongoing_projects/jiminy-project/miniwob-plusplus/html/miniwob/click-button.html"])
    element = wobenv.web_driver_list[0].driver.find_element_by_id("sync-task-cover")
    rect = element.rect
    x = rect['x'] + (rect['width'] / 2.)
    y = rect['y'] + (rect['height'] / 2.)
    vnc_click = vnc_event.PointerEvent(x, y, 1)
    wobenv.step([vnc_click])
    vnc_click = vnc_event.PointerEvent(x, y, 0)
    obs, _, _, _ = wobenv.step([vnc_click])
