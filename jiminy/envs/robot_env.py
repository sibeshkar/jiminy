from jiminy import vectorized

import logging
import numpy as np
import os

import getpass
import random
import uuid

from jiminy.gym.utils import reraise

from jiminy import error, pyprofile, rewarder, spaces, twisty, vectorized
from jiminy import remotes as remotes_module
from jiminy.envs import diagnostics
from jiminy.remotes import healthcheck
from jiminy.runtimes import registration

logger = logging.getLogger(__name__)

def default_client_id():
    return '{}-{}'.format(uuid.uuid4(), getpass.getuser())

def rewarder_session(which):
    if which is None:
        which = rewarder.RewarderSession

    if isinstance(which, type):
        return which
    else:
        raise error.Error('Invalid RewarderSession driver: {!r}'.format(which))

def build_observation_n(info_n):
    observation_n = []
    for info in info_n:
        dom = info.pop('rewarder.observation', [])
        obs = {
            'dom': dom,
        }
        if 'env.generic' in info:
            obs['generic'] = info.pop('env.generic')
        observation_n.append(obs)
    return observation_n

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

class RobotEnv(vectorized.Env):
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
        self.observation_space = spaces.VNCObservationSpace()
        self.action_space = spaces.VNCActionSpace()
        self._send_actions_over_websockets = True
        self._skip_network_calibration = True

    def configure(self, env=None, task=None, remotes=None, #TODO: Currently env and task is singular, will be plural in the future
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
        
        runtime = 'world-of-bits' #TODO:Remove this in future editions 

        twisty.start_once()

        if remotes is None:
            remotes = os.environ.get('GYM_VNC_REMOTES', '1')

        if client_id is None:
            client_id = default_client_id()

            

        logger.info("Configuring the environment...")

        self.remote_manager, self.n = remotes_module.build(
            client_id=client_id,
            remotes=remotes, runtime=runtime, start_timeout=start_timeout,
            api_key=api_key,
            use_recorder_ports=record,
            env=env,
            task=task
        )

        self.connection_names = [None] * self.n
        self.connection_labels = [None] * self.n

        self.crashed = {}

        self.allow_reconnect = replace_on_crash and self.remote_manager.supports_reconnect

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

        # if self._send_actions_over_websockets:
        #     self.rewarder_session.send_action(action_n, self.spec.id)

        if self.rewarder_session:
            reward_n, done_n, info_n, err_n = self._pop_rewarder_session(peek_d)
        else:
            reward_n = done_n = [None] * self.n
            info_n = [{} for _ in range(self.n)]
            err_n = [None] * self.n


        
        observation_n = build_observation_n(info_n)

        self._handle_initial_n(observation_n, reward_n)
        #self._handle_err_n(err_n, info_n, observation_n, reward_n, done_n)
        #self._handle_crashed_n(info_n)


        return observation_n, reward_n, done_n, {'n': info_n}
    
    def _pop_rewarder_session(self, peek_d):
        with pyprofile.push('vnc_env.VNCEnv.rewarder_session.pop'):
            reward_d, done_d, info_d, err_d = self.rewarder_session.pop(peek_d=peek_d)
        #print("Info obtained is", info_d)
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
                rewarder_address=remote.rewarder_address, rewarder_password=remote.rewarder_password, env=remote.env, task=remote.task)

    def _action_d(self, action_n):
        action_d = {}
        for i, action in enumerate(action_n):
            action_d[self.connection_names[i]] = action
        return action_d

    def connect(self, i, name, vnc_address, rewarder_address,env, task=None, vnc_password=None, rewarder_password=None ):
        self.connection_names[i] = name
        self.connection_labels[i] = '{}'.format(name)
        
        if self.rewarder_session is not None:
            # if self.spec is not None:
            #     env_id = self.spec.id
            # else:
            #     env_id = None

            env_id = env #temporarily created, not will pass env_id as argument finally
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
                seed=seed, env_id=env_id, task=task,
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
            if self.diagnostics:
                self.diagnostics.close(i)
            #self.mask.close(i)
            self.connection_names[i] = None
            self.connection_labels[i] = None
        else:
            if hasattr(self, 'rewarder_session') and self.rewarder_session:
                self.rewarder_session.close()
            if hasattr(self, 'diagnostics') and self.diagnostics:
                self.diagnostics.close()
            if hasattr(self, 'remotes_manager') and self._remotes_manager:
                self._remotes_manager.close()

    def _render(self, mode='human', close=False):
        pass

    def _handle_initial_n(self, observation_n, reward_n):
        if self.rewarder_session is None:
            return

        for i, reward in enumerate(reward_n):
            if reward is None:
                # Index hasn't come up yet, so ensure the observation
                # is blanked out.
                observation_n[i] = None

    def _handle_err_n(self, err_n, info_n, observation_n=None, reward_n=None, done_n=None):
        # Propogate any errors upwards.
        for i, err in enumerate(zip(err_n)):
            if err is None:
                # All's well at this index.
                continue

            if observation_n is not None:
                observation_n[i] = None
                done_n[i] = True

            # Propagate the error
            if err is not None:
                info_n[i]['error'] = 'Rewarder session failed: {}'.format(err)

            #extra_logger.info('[%s] %s', self.connection_names[i], info_n[i]['error'])

            if self.allow_reconnect:
                logger.info('[%s] Making a call to the allocator to replace crashed index: %s', self.connection_names[i], info_n[i]['error'])
                self.remote_manager.allocate([str(i)])

            self.crashed[i] = self.connection_names[i]
            self._close(i)
    
    def _handle_crashed_n(self, info_n):
        # for i in self.crashed:
        #     info_n[i]['env_status.crashed'] = True

        if self.allow_reconnect:
            return

        if len(self.crashed) > 0:
            errors = {}
            for i, info in enumerate(info_n):
                if 'error' in info:
                    errors[self.crashed[i]] = info['error']

            if len(errors) == 0:
                raise error.Error('{}/{} environments have crashed. No error key in info_n: {}'.format(len(self.crashed), self.n, info_n))
            else:
                raise error.Error('{}/{} environments have crashed! Most recent error: {}'.format(len(self.crashed), self.n, errors))

    
    def __str__(self):
        return 'RobotEnv'