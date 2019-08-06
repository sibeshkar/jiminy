import tensorflow as tf
import numpy as np

class FileInitializer(tf.keras.initializers.Initializer):

    def __init__(self, filename, verify_shape=True):
        self.fname = filename
        self.verify_shape = verify_shape

    def __call__(self):
        weights = np.load(self.fname)
        tensor = tf.Constants(weights)
        return tensor

    def get_config(self):
        config = dict()
        config["fname"] = self.fname
        config["verify_shape"] = verify_shape
        return config

    @classmethod
    def from_config(cls, dictv):
        return cls(dictv["fname"], dictv["verify_shape"])
