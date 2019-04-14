#!/usr/bin/env python
import argparse
import logging
import sys
import time

from lib import wob_vnc

from PIL import Image

REMOTES_COUNT = 1

import gym
import jiminy # register the universe environments

from jiminy import wrappers

logger = logging.getLogger()

def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    args = parser.parse_args()

    if args.verbosity == 0:
        logger.setLevel(logging.INFO)
    elif args.verbosity >= 1:
        logger.setLevel(logging.DEBUG)


    env = gym.make('wob.mini.ClickDialog-v0')
    env = jiminy.wrappers.experimental.SoftmaxClickMouse(env)
    env = wob_vnc.MiniWoBCropper(env)
    wob_vnc.configure(env, wob_vnc.remotes_url(port_ofs=0, hostname='localhost', count=REMOTES_COUNT))  # automatically creates a local docker container
    
    observation_n = env.reset()
    idx = 0
    while True:
        # your agent here
        #
        # Try sending this instead of a random action: ('KeyEvent', 'ArrowUp', True)
        action_n = [env.action_space.sample() for ob in observation_n]
        observation_n, reward_n, done_n, info = env.step(action_n)
        print("idx: {}, reward: {}".format(idx*REMOTES_COUNT, reward_n))
        idx += 1
    return 0

if __name__ == '__main__':
    sys.exit(main())