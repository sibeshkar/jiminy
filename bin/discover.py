#!/usr/bin/env python
import argparse
import logging
import json
import sys

from jiminy import kube

logger = logging.getLogger()

def discover_batches(args):
    batches = kube.discover_batches()
    for name, info in batches.items():
        print('{}: count={}'.format(name, info['count']))

def discover(args):
    pods = kube.discover(args.batch)

    if args.repr:
        pods = [pod for pod in pods if pod['ready']]
        print(repr(pods))
    else:
        print(json.dumps(pods))

def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')

    subparsers = parser.add_subparsers()
    sub = subparsers.add_parser('discover_batches')
    sub.set_defaults(func=discover_batches)

    sub = subparsers.add_parser('discover')
    sub.set_defaults(func=discover)
    sub.add_argument('batch', help='The ID of the batch.')
    sub.add_argument('-r', '--repr', action='store_true', help='Dump a repr of the list, with any non-ready pods filtered out, suitable for pasting into a script.')

    args = parser.parse_args()

    if args.verbosity == 0:
        logger.setLevel(logging.INFO)
    elif args.verbosity >= 1:
        logger.setLevel(logging.DEBUG)

    args.func(args)

    return 0

if __name__ == '__main__':
    sys.exit(main())
