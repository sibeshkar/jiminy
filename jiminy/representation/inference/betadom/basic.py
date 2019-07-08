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
from data import create_dataset
from jiminy.utils.ml import Vocabulary
import tensorflow as tf
tf.enable_eager_execution()

class BaseModel():
    def __init__(self, max_length=10,
            tag_vocab_size=50,
            screen_shape=(160,210)):
        self.max_length = max_length
        self.tag_vocab_size = tag_vocab_size
        self.screen_shape = screen_shape

    def create_model(self):
        w,h = self.screen_shape
        self.image_input = tf.keras.Input(shape=(w,h,3), dtype=tf.float32) # image input is 256x256 RGB
        self.image_model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (3,3), padding='valid', activation='relu'),
            tf.keras.layers.Conv2D(16, (3,3), padding='same', strides=2, activation='relu'),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', strides=2, activation='relu'),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', strides=2, activation='relu'),
            tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.RepeatVector(self.max_length)
            ])
        encoded_image = self.image_model(self.image_input)

        self.tag_input = tf.keras.Input(shape=(self.max_length,))
        self.language_model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.tag_vocab_size, 64, input_length=self.max_length),
            tf.keras.layers.CuDNNLSTM(128, return_sequences=True),
            tf.keras.layers.CuDNNLSTM(128, return_sequences=True)
            ])
        encoded_tag = self.language_model(self.tag_input)

        decoder_input = tf.concat([encoded_image, encoded_tag], axis=-1)
        self.decoder_model = tf.keras.Sequential([
            tf.keras.layers.CuDNNLSTM(256, return_sequences=True),
            tf.keras.layers.CuDNNLSTM(256, return_sequences=False),
            ])
        decoder_model_output = self.decoder_model(decoder_input)

        tag_bounding_box = tf.keras.layers.Dense(4, activation='sigmoid')(decoder_model_output)
        tag_output = tf.keras.layers.Dense(self.tag_vocab_size, activation='softmax')(decoder_model_output)
        self.decoder_output = tf.concat([tag_output, tag_bounding_box], axis=-1)
        self.model = tf.keras.Model(inputs=[self.image_input, self.tag_input],
                outputs=[tag_bounding_box, tag_output])

    def forward_pass(self, img):
        img_data = np.array(img).astype(np.float32).reshape([1] + list(self.screen_shape))
        target_data = np.array([0 for _ in range(self.max_length)]).reshape((1, self.max_length))
        object_list = []
        multiplier = list(self.screen_shape) + list(self.screen_shape)

        while not is_done:
            softmax, bb = self.model([img_data, target_data])
            bb = bb.numpy() * multiplier
            object_type_list.append((np.argmax(softmax.numpy()), bb))

        return []

if __name__ == "__main__":
    baseModel = BaseModel(screen_shape=(300,300))
    baseModel.create_model()
    print(baseModel.model.summary())

    vocab = Vocabulary(["text","input", "checkbox", "button", "click"])
    dataset = create_dataset("logdir", 32, vocab, baseModel.max_length, tuple(baseModel.screen_shape + [3])

    for (img, prev_tags, _) in dataset.take(4):
        img = tf.cast(img, dtype=tf.float32)
        val = baseModel.model([img, prev_tags])
        print(val, [v.shape for v in val])
