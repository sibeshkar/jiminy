#!/usr/bin/env python
import argparse
import logging
import sys

from twisted.internet import endpoints, protocol, reactor

# In modules, use `logger = logging.getLogger(__name__)`
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stderr))

class VNCProxyServer(protocol.Protocol, object):
    def __init__(self):
        self._broken = False
        self.vnc_client = None

        self.to_server = open('to_server', 'w')
        self.from_server = open('from_server', 'w')

    def connectionMade(self):
        logger.debug('Connection received from VNC client')
        factory = protocol.ClientFactory()
        factory.protocol = VNCProxyClient
        factory.vnc_server = self
        endpoint = endpoints.clientFromString(reactor, self.factory.vnc_address)

        def _connect_callback(client):
            if self._broken:
                client.transport.loseConnection()
            self.vnc_client = client
        def _connect_errback(reason):
            logger.error('[VNCProxyServer] Connection failed: %s', reason)
            self.transport.loseConnection()
        endpoint.connect(factory).addCallbacks(_connect_callback, _connect_errback)

    def connectionLost(self, reason):
        logger.debug('Losing connection from VNC server')
        if self.vnc_client:
            self.vnc_client.transport.loseConnection()

    def dataReceived(self, data):
        if not self.vnc_client:
            logger.error('Bytes received from client before connecting to server')
            self._broken = True
            self.transport.loseConnection()

        self.from_server.write(data)
        self.vnc_client.sendData(data)

    def sendData(self, data):
        logger.debug('Proxy message to VNC client: %s', data)
        self.to_server.write(data)
        self.transport.write(data)

class VNCProxyClient(protocol.Protocol, object):
    def __init__(self):
        self.vnc_server = None

    def connectionMade(self):
        logger.debug('Connection made to VNC server')
        self.vnc_server = self.factory.vnc_server

    def connectionLost(self, reason):
        logger.debug('Losing connection to VNC server')
        if self.vnc_server:
            self.vnc_server.transport.loseConnection()

    def dataReceived(self, data):
        logger.debug('Proxy message from VNC server: %s', data)
        self.vnc_server.sendData(data)

    def sendData(self, data):
        logger.debug('Proxy message to VNC server: %s', data)
        self.transport.write(data)

def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('-l', '--listen-address', default='0.0.0.0:5898', help='Address to listen on')
    parser.add_argument('-s', '--vnc-address', default='127.0.0.1:5900', help='Address of the VNC server to run on.')
    args = parser.parse_args()

    if args.verbosity == 0:
        logger.setLevel(logging.INFO)
    elif args.verbosity >= 1:
        logger.setLevel(logging.DEBUG)

    factory = protocol.ServerFactory()
    factory.protocol = VNCProxyServer
    factory.vnc_address = 'tcp:{}'.format(args.vnc_address)

    host, port = args.listen_address.split(':')
    port = int(port)

    logger.info('Listening on %s:%s', host, port)
    reactor.listenTCP(port, factory, interface=host)
    reactor.run()
    return 0

if __name__ == '__main__':
    sys.exit(main())
