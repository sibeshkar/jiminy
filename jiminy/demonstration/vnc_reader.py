import itertools
import json
import logging
import os
import random

import time
from jiminy.gym import error

logger = logging.getLogger(__name__)


class VNCReader():
    """
    Iterator that replays a VNCDemonstration at a certain FPS.
    """

    def __init__(self, fps = 20, demo_file='demo.rbs'):

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        observation, reward, done, info, action = next(self.framed_event_reader)

        return observation, reward, done, info, action