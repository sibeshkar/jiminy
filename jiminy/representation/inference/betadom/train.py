"""
Training algorithm is simple. Backprop through time.
"""

from jiminy.representation.inference.betadom.data import create_dataset
from jiminy.utils.ml import Vocabulary, ScreenVisualizerCls
import tensorflow as tf
tf.enable_eager_execution()
from jiminy.representation.inference.betadom.basic import BaseModel
import datetime

class PrettyPrintOutput(tf.keras.Callback):
    def __init__(self, dataset, vocab, logdir, prefix):
        super(PrettyPrintOutput, self).__init__()
        self.dataset = dataset
        self.vocab = vocab

        self.logdir, self.prefix = logdir, prefix
        self.screenVisualizer = ScreenVisualizerCls(self.logdir, self.prefix)

    def on_epoch_end(self, epoch, logs=None):
        for i in self.dataset:
            label, bounding_box = self.model.forward_pass(self.dataset[i])
            self.screenVisualizer(self.dataset[i], bounding_box, label, epoch)


class BaseModelTrainer(object):
    def __init__(self, learning_rate=1e-4, learning_algorithm="Adam",
            model_dir="logdir", vocab=Vocabulary(["START", "END", "input"]),
            lambda_bb=1e-2, num_gpus=2):
        """
        Learning algorithm must be a valid one
        model_dir: wrt to the JIMINY_BASEROOT env variable
        """
        if learning_algorithm == "Adam":
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.lambda_bb = lambda_bb
        self.vocab = vocab

        self.dataset = create_dataset(model_dir, 64, vocab, 10, (300, 300, 3), int(1e5))
        self.baseModel = BaseModel(screen_shape=(300, 300), vocab=vocab)
        with tf.device("/cpu:0"):
            self.baseModel.create_model()
        self.baseModel.model = tf.keras.multi_gpu_model(self.baseModel.model)
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
                1e-2,
                1.
            ]

    def get_metric(self):
        return tf.keras.metrics.MeanSquaredError()

if __name__ == "__main__":
    vocab = Vocabulary(["START", "text","input", "checkbox", "button", "click", "END"])
    bmt = BaseModelTrainer(vocab=vocab)

    callbacks = [tf.keras.callbacks.TensorBoard(log_dir="logs/BaseModel-{}".format(datetime.datetime.now().strftime("%d-%b-%Y::%H-%M"))
        ,update_freq=10),
            tf.keras.callbacks.ModelCheckpoint("logs/baseModel.tf", save_weights_only=False, load_weights_on_restart=True),
            ]
    bmt.train(epochs=100, callbacks=callbacks, steps_per_epoch=int(7185)) # TODO(prannayk): automate setting up steps_per_epoch
