#!/usr/bin/env python
import argparse
import jiminy.gym as gym
import jiminy
import logging
import numpy as np
import sys

# In modules, use `logger = logging.getLogger(__name__)`
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stderr))

def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('-o', '--output', required=True, help='Where to save trace.')
    parser.add_argument('-e', '--env-id', default='Pong-v3', help='Which env to run.')
    parser.add_argument('-s', '--vnc-address', default='127.0.0.1:5900', help='Address of the VNC server to run on.')
    parser.add_argument('-r', '--rewarder-address', default='127.0.0.1:15900', help='Address of the rewarder server to run on.')
    parser.add_argument('-S', '--seed', type=int, default=0, help='Set seed.')

    args = parser.parse_args()

    if args.verbosity == 0:
        logger.setLevel(logging.INFO)
    elif args.verbosity >= 1:
        logger.setLevel(logging.DEBUG)

    observations = []

    vnc = args.env_id.startswith('VNC')
    env = gym.make(args.env_id)
    if args.seed is not None:
        env.seed(args.seed)
    if vnc:
        env.configure(vnc_address=args.vnc_address, rewarder_address=args.rewarder_address)
        noop = []
    else:
        assert env.get_action_meanings()[0] == 'NOOP'
        noop = 0

    ob = env.reset()
    observations.append(ob)

    for i in range(100):
        ob, reward, done, info = env.step(noop)
        observations.append(ob)

    np.save(args.output, observations)

    return 0

if __name__ == '__main__':
    sys.exit(main())
