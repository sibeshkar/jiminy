"""
Basic model for creating betaDOM from Pixels

Based on:
    1. Pix2Code: https://github.com/tonybeltramelli/pix2code
    2. Screenshot-to-code: https://github.com/emilwallner/Screenshot-to-code

The basic model takes as input:
    1. Screenshot -- pixel image
    2. Set of previously created objects from the screenshot
Outputs:
    1. A single JiminyBaseObject's: objectType
    2. Same single JiminyBaseObject's: boundingBox
"""
from jiminy.representation.inference.betadom.data import create_dataset
from jiminy.utils.ml import Vocabulary
import tensorflow as tf
import numpy as np
import time
# tf.enable_eager_execution()

class BaseModel():
    def __init__(self, max_length=10,
            vocab=None,
            screen_shape=(160,210),
            config=dict()):
        self.max_length = max_length
        self.vocab = vocab
        self.tag_vocab_size = vocab.length
        self.screen_shape = screen_shape

        self.config = config
        if not "last_conv" in self.config:
            self.config["last_conv"] = 96
        if not "lm_lstm_size" in self.config:
            self.config["lm_lstm_size"] = 128
        if not "lm_lstm_layer_num" in self.config:
            self.config["lm_lstm_layer_num"] = 2
        if not "decoder_middle_lstm" in self.config:
            self.config["decoder_middle_lstm"] = True



    def create_model(self):
        w,h = self.screen_shape
        self.image_input = tf.keras.Input(shape=(w,h,3), dtype=tf.float32) # image input is 256x256 RGB
        self.image_model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (5,5), padding='valid', activation='relu'),
            tf.keras.layers.Conv2D(16, (3,3), padding='same', strides=2, activation='relu'),
            tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', strides=4, activation='relu'),
            tf.keras.layers.Conv2D(64, (5,5), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', strides=4, activation='relu'),
            tf.keras.layers.Conv2D(self.config["last_conv"], (5,5), padding='same', activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.RepeatVector(self.max_length)
            ])
        encoded_image = self.image_model(self.image_input)

        self.tag_input = tf.keras.Input(shape=(self.max_length,))
        self.language_model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.tag_vocab_size, 64, input_length=self.max_length)
            ])
        for _ in range(self.config["lm_lstm_layer_num"]):
            self.language_model.add(tf.keras.layers.LSTM(128, return_sequences=True))
        encoded_tag = self.language_model(self.tag_input)

        decoder_input = tf.keras.layers.concatenate(inputs=[encoded_image, encoded_tag], axis=-1)
        self.decoder_model = tf.keras.Sequential()
        if self.config["decoder_middle_lstm"]:
            self.decoder_model.add(tf.keras.layers.LSTM(self.config["lm_lstm_size"], return_sequences=True))
        self.decoder_model.add(tf.keras.layers.LSTM(self.config["lm_lstm_size"], return_sequences=False))
        decoder_model_output = self.decoder_model(decoder_input)

        tag_bounding_box = tf.keras.layers.Dense(4, activation='sigmoid')(decoder_model_output) * tf.convert_to_tensor([w, h, w, h], dtype=tf.float32)
        tag_output = tf.keras.layers.Dense(self.tag_vocab_size, activation='softmax')(decoder_model_output)
        self.decoder_output = tf.keras.layers.concatenate(inputs=[tag_output, tag_bounding_box], axis=-1)
        self.model = tf.keras.Model(inputs=[self.image_input, self.tag_input],
                outputs=[tag_bounding_box, tag_output])


    def forward_pass(self, img):
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
        img_tensor = tf.reshape(img_tensor, shape=[1] + list(img.shape))

        tag_list = ["START" for _ in range(self.max_length)]
        bb_list = []
        count = 0
        while (tag_list[-1] != "END") and count < self.max_length:
            tag_list_sym = self.vocab.to_sym(tag_list)
            tags = tf.reshape(tf.convert_to_tensor(tag_list_sym, dtype=tf.int64), [1, self.max_length])
            bb,tag = self.model([img_tensor, tags])
            bb_list.append(bb.numpy()[0])
            last_tag = self.vocab.to_key(np.argmax(tag.numpy(), axis=-1))
            tag_list = tag_list[1:] + last_tag
            count += 1

        if tag_list[-1] == "END":
            n = len(tag_list)
            tag_list = tag_list[:(n-1)]
            bb_list = bb_list[:(n-1)]
        i = 0
        while i < len(tag_list) and tag_list[i] == "START": i+=1
        tag_list = tag_list[i:]
        bb_list = bb_list[i:]

        return tag_list, bb_list

    def get_loss(self):
        return ["mse", "categorical_crossentropy"]

    def get_loss_weights(self):
        return [1e-1, 0.5]

    def get_metric(self):
        return [["mae"], ["accuracy"]]

if __name__ == "__main__":
    vocab = Vocabulary(["START", "text","input", "checkbox", "button", "click", "END"])
    baseModel = BaseModel(screen_shape=(300,300), vocab=vocab)
    baseModel.create_model()
    print(baseModel.model.summary())

    dataset = create_dataset("logdir", 32, vocab, 10, (300, 300,3))

    for (X, y) in dataset.take(1):
        print(X)
        val = baseModel.model(X)
        print([v.shape for v in val])

    tstart = time.time()
    img = np.zeros([300, 300,3], np.float32)

    tstart = time.time()
    tag_list = baseModel.forward_pass(img)
    print(tag_list, time.time() - tstart)

