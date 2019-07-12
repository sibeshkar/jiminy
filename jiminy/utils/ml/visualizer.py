"""
Responsible for handling visualization of screens
during training of model to understand how well it works
"""
import cv2
# import tensorflow as tf

class ScreenVisualizerCls():
    def __init__(self, logdir, prefix):
        self.logdir = logdir
        self.prefix = prefix

    def __call__(self, img, bounding_box, label, epoch):
        for i, bb in enumerate(bounding_box):
            cv2.rectangle(img, bb[0], bb[1], bb[2], bb[3])
        cv2.imwrite("{}/{}-epoch-{}.png".format(self.logdir, self.prefix, self.epoch), img)
