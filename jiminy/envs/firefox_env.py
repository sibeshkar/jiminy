from jiminy.envs import DummyVNCEnv

class SeleniumEnv(DummyVNCEnv):
    """
    Selenium based Env to interact with web pages
    """
    def __init__(self, ):
        self._started = False
        self.observation_space = spaces.VNCObservationSpace()
        self.action_space = spaces.VNCActionSapce()

    def configure(self, remotes=None,
                   client_id=None,
                   start_timeout=None, docker_image=None,
                   ignore_clock_skew=False, disable_action_probes=False,
                   vnc_driver=None, vnc_kwargs={},
                   replace_on_crash=False, allocate_sync=True,
                   observer=False,
                   _n=3,
        ):
        self.n = _n
        self.started = True

    def _reset(self):
        return [None] * self.n

    def _step(self, action_n):
        assert self.n == len(action_n), "Expected {} actions but received {}: {}".format(self.n, len(action_n), action_n)

