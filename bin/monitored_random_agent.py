#!/usr/bin/env python
import argparse
import json
import logging
import numpy as np
import os
import time

import jiminy.gym as gym

from jiminy import error, kube, vnc

logger = logging.getLogger()

# The world's simplest agent!
class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def __call__(self, observation, reward, done):
        return self.action_space.sample()

if __name__ == '__main__':
    # You can optionally set up the logger. Also fine to set the level
    # to logging.DEBUG or logging.WARN if you want to change the
    # amount of output.
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-e', '--env_id', default='VNCSpaceInvaders-v3', help='Which environment to run on.')
    parser.add_argument('-r', '--rewarder-address', help='Address of the rewarder server to run on.')
    parser.add_argument('-s', '--vnc-address', help='Address of the VNC server to run on.')
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('-d', '--dummy', action='store_true', help='Do not submit actions')
    parser.add_argument('-R', '--render', action='store_true', help='Render the game')
    parser.add_argument('-b', '--benchmark-run-id', help='The BenchmarkRun to attach this to')
    args = parser.parse_args()

    if args.verbosity == 0:
        logger.setLevel(logging.INFO)
    elif args.verbosity >= 1:
        logger.setLevel(logging.DEBUG)

    env = gym.make(args.env_id)
    env.seed(0)
    env.configure(
        name='main',
        vnc_address=args.vnc_address,
        rewarder_address=args.rewarder_address,
    )

    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/vnc-random-agent-results/{}'.format(args.env_id)
    env.monitor.start(outdir, force=True)

    agent = RandomAgent(env.action_space)

    episode_count = 1
    max_steps = 400
    reward = 0
    done = False

    actions = 0
    start = time.time()

    synced = False
    while not env.screen_synced():
        logger.info('Waiting for screen to sync')
        time.sleep(0.25)
    logger.info('Screen synced!')

    for episodes in range(episode_count):
        start = time.time()
        observation = env.reset()
        logger.info('Reset duration: %s', time.time() - start)
        logger.debug('Initial observation: %s', observation)

        # Initialize the render while not realtime
        if args.render:
            env.render()

        target = time.time()
        for step in range(max_steps):
            target += 1/60.

            actions += 1
            if args.render:
                env.render()

            if args.dummy:
                action = []
            else:
                action = agent(observation, reward, done)

            # Take an action
            observation, reward, done, info = env.step(action)

            if (reward is not None and reward != 0) or done:
                logger.info('[%s] randomagent got: reward=%s done=%s', env.name, reward, done)

            if done:
                break

            delta = target - time.time()
            if delta > 0:
                time.sleep(delta)
            elif delta > 0.05:
                logger.info('Falling behind: %s', delta)
    env.monitor.close()

    # Upload to the scoreboard. We could also do this from another
    # process if we wanted.
    logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    gym.upload(outdir, benchmark_run_id=args.benchmark_run_id)
