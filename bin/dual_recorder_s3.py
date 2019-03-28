#!/usr/bin/env python
''' dual channel recorder (vnc + reward)
    the demo and rewards are uploaded to s3 automatically whenever the client disconnects
    to both vnc and reward.
'''
import argparse
import sys
import re
import logging

from autobahn.twisted import websocket
from twisted.internet import protocol

# somehow using reactor from twisted caused hanging behavior for reward proxy.
from twisted.internet import reactor
#from jiminy.twisty import reactor

from jiminy import utils
from jiminy.vncdriver.dual_proxy_server import DualProxyServer

logger = logging.getLogger()

if __name__ == '__main__':
    print('running dual recorder')
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('-l', '--vnc-listen-address', default='0.0.0.0:5899', help='VNC address to listen on')
    parser.add_argument('-s', '--vnc-address', default='127.0.0.1:5900', help='Address of the VNC server to run on.')
    parser.add_argument('-r', '--rewarder-listen-address', default='0.0.0.0:15899', help='Rewarder address to listen on')
    parser.add_argument('-t', '--rewarder-address', default='127.0.0.1:15900', help='Address of the reward server to run on.')
    parser.add_argument('-d', '--logfile-dir', default='/tmp/demo', help='Base directory to write logs for each connection')
    parser.add_argument('-b', '--bucket', default='boxware-vnc-demonstrations-dev', help='S3 bucket to upload demonstrations data to')
    args = parser.parse_args()

    if args.verbosity == 0:
        logger.setLevel(logging.INFO)
    elif args.verbosity >= 1:
        logger.setLevel(logging.DEBUG)

    # factory for VNC proxy server
    vnc_factory = protocol.ServerFactory()
    vnc_factory.protocol = DualProxyServer
    vnc_factory.vnc_address = 'tcp:{}'.format(args.vnc_address)
    vnc_factory.logfile_dir = args.logfile_dir
    vnc_factory.recorder_id = utils.random_alphanumeric().lower()
    vnc_factory.bucket = args.bucket
    vnc_factory.reward_proxy_bin = '/app/jiminy/bin/reward_recorder.py'

    vnc_host, vnc_port = args.vnc_listen_address.split(':')
    vnc_port = int(vnc_port)

    logger.info('[vnc_proxy] Listening on %s:%s', vnc_host, vnc_port)
    reactor.listenTCP(vnc_port, vnc_factory, interface=vnc_host)

    reactor.run()
