import logging

from jiminy.vncdriver.vnc_session import VNCSession
from jiminy.vncdriver.vnc_client import client_factory
from jiminy.vncdriver.screen import NumpyScreen, PygletScreen

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
