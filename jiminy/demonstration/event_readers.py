import heapq
import json
import logging

from jiminy import spaces

class ActionQueue(object):
    def __init__(self):
        self.actions = []
        self.pixel_format = []

    def key_event(self, key, down):
        event = spaces.KeyEvent(key, down)
        self.actions.append(event)

    def pointer_event(self, x, y, buttonmask):
        event = spaces.PointerEvent(x, y, buttonmask)
        self.actions.append(event)

    def set_pixel_format(self, server_pixel_format):
        self.pixel_format.append(server_pixel_format)

    def pop_all(self):
        output = self.actions, self.pixel_format
        self.actions = []
        self.pixel_format = []
        return output