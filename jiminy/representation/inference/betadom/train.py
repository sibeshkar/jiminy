"""
Training algorithm is simple. Backprop through time.
"""

from jiminy.representation.inference.betadom.data import create_dataset
import tensorflow as tf
tf.enable_eager_execution()
from jiminy.utils.ml import Vocabulary, ScreenVisualizerCallback, getVisualizationList
from jiminy.representation.inference.betadom.basic import BaseModel
import datetime
import argparse
import json
import os

class BaseModelTrainer(object):
    def __init__(self, learning_rate=1e-4, learning_algorithm="Adam",
            model_dir="logdir", vocab=Vocabulary(["START", "END", "input"]),
            lambda_bb=1e-2, num_gpus=2, config=dict()):
        """
        Learning algorithm must be a valid one
        model_dir: wrt to the JIMINY_BASEROOT env variable
        """
        if learning_algorithm == "Adam":
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.lambda_bb = lambda_bb
        self.vocab = vocab

        self.dataset = create_dataset(model_dir, 64, vocab, 10, (300, 300, 3), int(1e5))
        self.baseModel = BaseModel(screen_shape=(300, 300), vocab=vocab, config=config)
        # with tf.device("/cpu:0"):
        self.baseModel.create_model()
        # self.baseModel.model = tf.keras.utils.multi_gpu_model(self.baseModel.model, num_gpus)
        print("Created Model")

        self.baseModel.model.summary()
        self.baseModel.model.compile(optimizer=self.optimizer,
                loss=self.get_loss(),
                loss_weights=self.get_loss_weights(),
                metrics=[self.get_metric()])

    def train(self, epochs=100, callbacks=[], steps_per_epoch=100):
        self.dataset = self.dataset.repeat(epochs)
        self.baseModel.model.fit(self.dataset, epochs=epochs,
                callbacks=callbacks,
                steps_per_epoch=steps_per_epoch)

    def get_loss(self):
        return ["mse", "categorical_crossentropy"]

    def get_loss_weights(self):
        return [
                1.,
                0.5
            ]

    def get_metric(self):
        return tf.keras.metrics.MeanSquaredError()

parser = argparse.ArgumentParser(description="BaseModelTrainer settings")
parser.add_argument("--model_name", dest="model_name", action="store",
        default="baseModel.h5", help="Model name to which training has to be stored")
parser.add_argument("--test", dest="test", action="store_const",
        const=True, default=False, help="Run the model in test model on a small batch of samples")
parser.add_argument("--model_config", dest="model_config", action="store",
        default="model_config/small.json", help="Defines the basic model which is smaller than stored params")
args = parser.parse_args()

if __name__ == "__main__":
    start_time = datetime.datetime.now().strftime("%d-%b-%Y::%H-%M")
    vocab = Vocabulary(["START", "text","input", "checkbox", "button", "click", "END"])
    config_dict = dict()
    if os.path.exists(args.model_config):
        with open(args.model_config) as f:
            json_obj = json.load(f)
        config_dict = dict(json_obj)

    bmt = BaseModelTrainer(learning_rate=1e-4, vocab=vocab, config=config_dict)

    visualization_img_list = getVisualizationList(bmt.dataset)
    logdir = "./logdir"
    prefix = "{}".format(start_time)

    callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir="logs/BaseModel-{}".format(start_time), update_freq=10),
            tf.keras.callbacks.ModelCheckpoint("logs/{}".format(args.model_name), save_weights_only=True, load_weights_on_restart=True),
            ScreenVisualizerCallback(dataset=visualization_img_list, vocab=vocab, logdir=logdir, prefix=prefix, baseModel=bmt.baseModel)
            ]
    epochs, steps_per_epoch = 100, 7185
    if args.test:
        # test the training mechanism
        epochs, steps_per_epoch = 10, 10
    bmt.train(epochs=epochs, callbacks=callbacks, steps_per_epoch=int(steps_per_epoch))
