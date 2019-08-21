import logging
import os
import re
import six.moves.urllib.parse as urlparse

from jiminy import error, utils
from jiminy.remotes import remote

logger = logging.getLogger(__name__)

class HttpAddresses(object):
    @classmethod
    def build(cls, remotes, env, task, **kwargs):
        parsed = urlparse.urlparse(remotes)
        print(parsed.scheme)
        # if parsed.scheme != 'http' or parsed.scheme != 'https':
        #     raise error.Error('HttpAddresses must be initialized with a string starting with http:// {} {}'.format(remotes, parsed.scheme))

        addresses = parsed.netloc.split(',')
        query = urlparse.parse_qs(parsed.query)
        # We could support per-backend passwords, but no need for it
        # right now.
        password = query.get('password', [utils.default_password()])[0]
        rewarder_addresses = addresses
        res = cls(rewarder_addresses, vnc_password=password, rewarder_password=password, env=env, task=task, **kwargs)
        return res, res.available_n

    def __init__(self, rewarder_addresses, vnc_password, rewarder_password, env, task=None,start_timeout=None):
        if rewarder_addresses is not None:
            self.available_n = len(rewarder_addresses)
        else:
            assert False

        self.env = env
        self.task = task

        self.supports_reconnect = False
        self.connect_rewarder = rewarder_addresses is not None
        if rewarder_addresses is None:
            logger.info("No rewarder addresses were provided, so this env cannot connect to the remote's rewarder channel, and cannot send control messages (e.g. reset)")

        self.rewarder_addresses = rewarder_addresses
        self.rewarder_password = rewarder_password
        if start_timeout is None:
            start_timeout = 2 * self.available_n + 5
        self.start_timeout = start_timeout

        self._popped = False

    def pop(self, n=None):
        if self._popped:
            assert n is None
            return []
        self._popped = True

        remotes = []
        for i in range(self.available_n):

            if self.rewarder_addresses is not None:
                rewarder_address = self.rewarder_addresses[i]
            else:
                rewarder_address = None

            name = self._handles[i]
            env = remote.Remote(
                handle=self._handles[i],
                vnc_address=None,
                vnc_password=None,
                rewarder_address=rewarder_address,
                rewarder_password=self.rewarder_password,
                env=self.env,
                task=self.task
            )
            remotes.append(env)
        return remotes

    def allocate(self, handles, initial=False, params={}):
        if len(handles) > self.available_n:
            raise error.Error('Requested {} handles, but only have {} envs'.format(len(handles), self.available_n))
        self.n = len(handles)
        self._handles = handles

    def close(self):
        pass