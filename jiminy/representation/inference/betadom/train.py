"""
Training algorithm is simple. Backprop through time.
"""

from jiminy.representation.inference.betadom.data import create_dataset
from jiminy.utils.ml import Vocabulary
import tensorflow as tf
from jiminy.representation.inference.betadom.basic import BaseModel
import datetime

class BaseModelLoss(tf.keras.losses.Loss):
    def __init__(self, lambda_bb=1e-2):
        super(BaseModelLoss, self).__init__()
        self.lambda_bb = lambda_bb

    def call(self, y_true, y_pred):
        if (int(y_pred.shape[-1]) == 8):
            return tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        bb_prediction_error = tf.keras.losses.MSE(y_true, y_pred)
        return self.lambda_bb*bb_prediction_error

class BaseModelTrainer(object):
    def __init__(self, learning_rate=1e-4, learning_algorithm="Adam",
            model_dir="logdir", vocab=Vocabulary(["START", "END", "input"]),
            lambda_bb=1e-2):
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
        self.baseModel.create_model()
        print("Created Model")
        self.baseModel.model.summary()
        self.baseModel.model.compile(optimizer=self.optimizer,
                loss=self.get_loss(),
                loss_weights=self.get_loss_weights(),
                metrics=[self.get_metric()])

    def train(self, dataset, epochs=100, callbacks=[], steps_per_epoch=100):
        self.dataset = self.dataset.repeat(epochs)
        self.baseModel.model.fit(dataset, epochs=epochs,
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
    print(bmt.dataset)
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir="logs/BaseModel-{}".format(datetime.datetime.now().strftime("%d-%b-%Y::%H-%M"))
        ,update_freq=10)]
    bmt.train(dataset=bmt.dataset, epochs=100, callbacks=callbacks, steps_per_epoch=int(7185)) # TODO(prannayk): automate setting up steps_per_epoch
