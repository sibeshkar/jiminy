from jiminy import vectorized

import logging
import numpy as np
import os

import getpass
import random
import uuid

from jiminy.gym.utils import reraise

from jiminy import error, pyprofile, rewarder, spaces, twisty, vectorized, vncdriver
from jiminy import remotes as remotes_module
from jiminy.envs import diagnostics
from jiminy.remotes import healthcheck
from jiminy.runtimes import registration

# The Go driver is the most supported one. So long as the Go driver
# turns out to be easy to install, we'll continue forcing the Go
# driver here.
# noinspection PyUnresolvedReferences

logger = logging.getLogger(__name__)

def default_client_id():
    return '{}-{}'.format(uuid.uuid4(), getpass.getuser())

def go_vncdriver():
    import go_vncdriver
    return go_vncdriver.VNCSession

def rewarder_session(which):
    if which is None:
        which = rewarder.RewarderSession

    if isinstance(which, type):
        return which
    else:
        raise error.Error('Invalid RewarderSession driver: {!r}'.format(which))

def vnc_session(which=None):
    # Short circuit so long as we're forcing the Go driver. Other code
    # left behind for the future if we need to support the other
    # drivers again.
    if isinstance(which, type):
        # Used in the tests to pass a custom VNC driver
        return which

    logger.info('Using the golang VNC implementation')
    return go_vncdriver()

    if which is None:
        which = os.environ.get('JIMINY_VNCDRIVER')

    if isinstance(which, type):
        return which
    if which == 'go':
        logger.info('Using the golang VNC implementation')
        return go_vncdriver()
    elif which is None:
        try:
            go = go_vncdriver()
            logger.debug('Using golang VNC implementation')
            return go
        except ImportError as e:
            logger.info("Go driver failed to import: {}".format(e))
            logger.info("Using pure Python vncdriver implementation. Run 'pip install go-vncdriver' to install the more performant Go implementation. Optionally set the environment variable JIMINY_VNCDRIVER='go' to force its use.")
    else:
        raise error.Error('Invalid VNCSession driver: {!r}'.format(which))

def compile_action(event):
    if isinstance(event, tuple):
        if event[0] == 'KeyEvent':
            name, down = event[1:]
            return spaces.KeyEvent.by_name(name, down=down).compile()
        elif event[0] == 'PointerEvent':
            x, y, buttonmask = event[1:]
            return spaces.PointerEvent(x, y, buttonmask).compile()
    else:
        return event.compile()

class CoreVNCEnv(vectorized.Env):
    """

    >>> dummy_env = gym.make('VNC.Core-v0')
    >>> e = YourActionWrapper(dummy_env)
    >>> e = jiminy.wrappers.Unvectorize(e)
    >>> observation, reward, done, info = e.step(example_input_action)
    >>> assert observation['action'] == example_output_action

    """
    metadata = {
        'render.modes': ['human'], # we wrap with a Render which can render to rgb_array
        'semantics.async': True,
        'semantics.autoreset': True,
        'video.frames_per_second' : 60,
        'runtime.vectorized': True,
    }

    def __init__(self, fps=None, probe_key=None):
        self.metadata = dict(self.metadata)
        if fps is not None:
            self.metadata['video.frames_per_second'] = fps
        self._started = False
        self._remotes_manager = None

        self._probe_key = probe_key or 0xbeef1
        self._seed_value = None #non-random int given temporarily
        self.rewarder_session = None
        self.vnc_session = None
        self.observation_space = spaces.VNCObservationSpace()
        self.action_space = spaces.VNCActionSpace()
        self._send_actions_over_websockets = False
        self._skip_network_calibration = True

    def configure(self, envs=None, tasks=None, remotes=None,
                  client_id=None,
                  start_timeout=None, docker_image=None,
                  ignore_clock_skew=False, disable_action_probes=False,
                  vnc_driver=None, vnc_kwargs=None,
                  rewarder_driver=None,
                  replace_on_crash=False, allocate_sync=True,
                  observer=False, api_key=None,
                  record=False,
                  sample_env_ids=None,
    ):
        
        runtime = 'world-of-bits'

        twisty.start_once()

        if remotes is None:
            remotes = os.environ.get('GYM_VNC_REMOTES', '1')

        if client_id is None:
            client_id = default_client_id()

        if vnc_kwargs is None:
            vnc_kwargs = {}

        logger.info("Configuring the environment...")

        #self.remote_manager, self.n = remotes_module.build(client_id=client_id,remotes=remotes)
        self.remote_manager, self.n = remotes_module.build(
            client_id=client_id,
            remotes=remotes, runtime=runtime, start_timeout=start_timeout,
            api_key=api_key,
            use_recorder_ports=record,
        )

        self.connection_names = [None] * self.n
        self.connection_labels = [None] * self.n

        self.crashed = {}

        self.allow_reconnect = replace_on_crash and self.remote_manager.supports_reconnect

        if self.remote_manager.connect_vnc:
            cls = vnc_session(vnc_driver)
            vnc_kwargs.setdefault('start_timeout', self.remote_manager.start_timeout)
            if runtime == 'gym-core':
                vnc_kwargs.setdefault('encoding', 'zrle')
            else:
                vnc_kwargs.setdefault('encoding', 'tight')
                vnc_kwargs.setdefault('fine_quality_level', 50)
                vnc_kwargs.setdefault('subsample_level', 2)
            # Filter out None values, since some drivers may not handle them correctly
            vnc_kwargs = {k: v for k, v in vnc_kwargs.items() if v is not None}
            #logger.info('Using VNCSession arguments: %s. (Customize by running "env.configure(vnc_kwargs={...})"', vnc_kwargs)
            self.vnc_kwargs = vnc_kwargs
            self.vnc_session = cls()
        else:
            self.vnc_session = None

        #self._started = True

        self._observer = observer
        if self.remote_manager.connect_rewarder:
            cls = rewarder_session(rewarder_driver)
            self.rewarder_session = cls()
        else:
            self.rewarder_session = None

        self.remote_manager.allocate([str(i) for i in range(self.n)], initial=True)

        if self.rewarder_session or ignore_clock_skew:
            # Don't need rewarder session if we're ignoring clock skew
            if self.spec is not None:
                metadata_encoding = self.spec.tags.get('metadata_encoding')
            else:
                metadata_encoding = None
            self.diagnostics = diagnostics.Diagnostics(self.n, self._probe_key, ignore_clock_skew, metadata_encoding=metadata_encoding, disable_action_probes=disable_action_probes)
        else:
            self.diagnostics = None

        self._sample_env_ids = sample_env_ids
        
        self._started = True
        if allocate_sync:
            # Block until we've fulfilled n environments
            self._handle_connect(n=self.n)
        else:
            # Handle any backends which synchronously fufill their
            # allocation.
            self._handle_connect()

    def _reset(self):
        self._handle_connect()
        return [None] * self.n

    def _step(self, action_n):
        self._handle_connect()

        assert self.n == len(action_n), "Expected {} actions but received {}: {}".format(self.n, len(action_n), action_n)

        action_n, peek_d = self._compile_actions(action_n)

        if self.rewarder_session:
            reward_n, done_n, info_n, err_n = self._pop_rewarder_session(peek_d)
        else:
            reward_n = done_n = [None] * self.n
            info_n = [{} for _ in range(self.n)]
            err_n = [None] * self.n


        if self.vnc_session:
            # if self.diagnostics:
            #     self.diagnostics.clear_probes_when_done(done_n)
            #     self.diagnostics.add_probe(action_n, action_mask)
            action_d = self._action_d(action_n)

            visual_observation_n, obs_info_n, vnc_err_n = self._step_vnc_session(action_d)
            # Merge in any keys from the observation
            #self._propagate_obs_info(info_n, obs_info_n)
        else:
            visual_observation_n = [None] * self.n
            vnc_err_n = [None] * self.n

        

        # observation_n = [{
        #     'vision': np.zeros((1024, 768, 3), dtype=np.uint8),
        #     'text': [],	//http://127.0.0.1:3000/miniwob/bisect-angle.html

        #     'action': action_n[i]
        # } for i in range(self.n)]

        # reward_n = []
        # done_n = []
        # info_n = []
        # for reward_buffer in self._reward_buffers:
        #     reward, done, info = reward_buffer.pop()
        #     reward_n.append(reward)
        #     done_n.append(done)
        #     info_n.append(info)
        return visual_observation_n, reward_n, done_n, {'n': info_n}
    
    def _pop_rewarder_session(self, peek_d):
        with pyprofile.push('vnc_env.VNCEnv.rewarder_session.pop'):
            reward_d, done_d, info_d, err_d = self.rewarder_session.pop(peek_d=peek_d)

        reward_n = []
        done_n = []
        info_n = []
        err_n = []
        for name in self.connection_names:
            reward_n.append(reward_d.get(name, 0))
            done_n.append(done_d.get(name, False))
            info_n.append(info_d.get(name, {'env_status.disconnected': True}))
            err_n.append(err_d.get(name))
        return reward_n, done_n, info_n, err_n
    
    def _step_vnc_session(self, compiled_d):
        if self._send_actions_over_websockets:
            self.rewarder_session.send_action(compiled_d, self.spec.id)
            vnc_action_d = {}
        else:
            vnc_action_d = compiled_d

        with pyprofile.push('vnc_env.VNCEnv.vnc_session.step'):
            observation_d, info_d, err_d = self.vnc_session.step(vnc_action_d)

        observation_n = []
        info_n = []
        err_n = []
        for name in self.connection_names:
            observation_n.append(observation_d.get(name))
            info_n.append(info_d.get(name))
            err_n.append(err_d.get(name))

        return observation_n, info_n, err_n

    def _handle_connect(self, n=None):
        # Connect to any environments which are ready
        for remote in self.remote_manager.pop(n=n):
            if remote.name is not None:
                name = '{}:{}'.format(remote.handle, remote.name)
            else:
                name = remote.handle
            self.connect(
                int(remote.handle), name=name,
                vnc_address=remote.vnc_address, vnc_password=remote.vnc_password,
                rewarder_address=remote.rewarder_address, rewarder_password=remote.rewarder_password)

    def _action_d(self, action_n):
        action_d = {}
        for i, action in enumerate(action_n):
            action_d[self.connection_names[i]] = action
        return action_d

    def connect(self, i, name, vnc_address, rewarder_address, vnc_password=None, rewarder_password=None):
        self.connection_names[i] = name
        self.connection_labels[i] = '{}:{}'.format(name, vnc_address)
        if self.vnc_session is not None:
            kwargs = {
                'name': name,
                'address': vnc_address,
                'password': vnc_password,
            }
            kwargs.update(self.vnc_kwargs)

            try:
                self.vnc_session.connect(**kwargs)
            except TypeError as e:
                reraise(suffix="(HINT: this error was while passing arguments to the VNCSession driver: {})".format(kwargs))

            # TODO: name becomes index:pod_id
            # TODO: never log index, just log name
        
        if self.rewarder_session is not None:
            # if self.spec is not None:
            #     env_id = self.spec.id
            # else:
            #     env_id = None

            env_id = 'sibeshkar/wob-v0/ClickShades' #temporarily created, not will pass env_id as argument finally
            #env_id = 'wob.mini.TicTacToe-v0'
            if self._seed_value is not None:
                # Once we use a seed, we clear it so we never
                # accidentally reuse the seed. Seeds are an advanced
                # feature and aren't supported by most envs in any
                # case.
                seed = self._seed_value[i]
                self._seed_value[i] = None
            else:
                seed = 0

            assert rewarder_password, "Missing rewarder password: {}".format(rewarder_password)
            network = self.rewarder_session.connect(
                name=name, address=rewarder_address,
                seed=seed, env_id=env_id,
                fps=self.metadata['video.frames_per_second'],
                password=rewarder_password,
                label=self.connection_labels[i],
                start_timeout=self.remote_manager.start_timeout,
                observer=self._observer,
                skip_network_calibration=self._skip_network_calibration
            )
        else:
            network = None
    
    def _compile_actions(self, action_n):
        compiled_n = []
        peek_d = {}
        try:
            for i, action in enumerate(action_n):
                compiled = []
                compiled_n.append(compiled)
                for event in action:
                    # Handle any special control actions
                    if event == spaces.PeekReward:
                        name = self.connection_names[i]
                        peek_d[name] = True
                        continue

                    # Do a generic compile
                    compiled.append(compile_action(event))
        except Exception as e:
            raise error.Error('Could not compile actions. Original error: {} ({}). action_n={}'.format(e, type(e), action_n))
        else:
            return compiled_n, peek_d

    def _close(self, i=None):
        if i is not None:
            name = self.connection_names[i]
            if self.rewarder_session:
                self.rewarder_session.close(name)
            if self.vnc_session:
                self.vnc_session.close(name)
            if self.diagnostics:
                self.diagnostics.close(i)
            #self.mask.close(i)
            self.connection_names[i] = None
            self.connection_labels[i] = None
        else:
            if hasattr(self, 'rewarder_session') and self.rewarder_session:
                self.rewarder_session.close()
            if hasattr(self, 'vnc_session') and self.vnc_session:
                self.vnc_session.close()
            if hasattr(self, 'diagnostics') and self.diagnostics:
                self.diagnostics.close()
            if hasattr(self, 'remotes_manager') and self._remotes_manager:
                self._remotes_manager.close()
    
    def __str__(self):
        return 'CoreVNCEnv'


class DummyVNCEnvFinal(vectorized.Env):
    def __init__():
        pass
    def _seed(self, seed):
        pass
    def configure(self):
        pass
    def connect(self):
        pass
    def _close(self):
        pass
    def _reset(self):
        pass
    def _reset_mask(self):
        pass
    def _pop_rewarder_session(self, peek_d):
        pass
    def _step_vnc_session(self, compiled_d):
        pass
    def _compile_actions(self, action_n):
        pass
    def _action_d(self, action_n):
        pass
    def _step(self, action_n):
        pass
    def _handle_initial_n(self, observation_n, reward_n):
        pass
    def _handle_err_n(self, err_n, vnc_err_n, info_n, observation_n=None, reward_n=None, done_n=None):
        pass
    def _handle_connect(self, n=None):
        pass
    def _handle_crashed_n(self, info_n):
        pass
    def _handle_crashed_n(self, info_n):
        pass
    def _render(self, mode='human', close=False):
        pass
    def __str__(self):
        pass
    
    
    

    
    