import tensorflow as tf
import numpy as np
# tf.enable_eager_execution()

from file_initializer import FileInitializer

class Vocabulary(object):
    def __init__(self, objectlist):
        self.key2sym = dict()
        self.sym2key = dict()
        for i, obj in enumerate(objectlist):
            self.key2sym[obj] = i
            self.sym2key[i] = obj
        self.length = len(objectlist)

class AttentionLayer(tf.keras.Model):
    def __init__(self, units):
        super(AttentionLayer, self).__init__()

class BetaDOMNet(object):
    def attention_layer(self, units, query, values):
        W1 = tf.keras.layers.Dense(units)
        W2 = tf.keras.layers.Dense(units)
        V = tf.keras.layers.Dense(1)
        time_axis = tf.expand_dims(query, 1)
        score = V(tf.nn.tanh(W1(time_axis)
            + W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(values*attention_weights, axis=1)
        return context_vector, attention_weights

    def __init__(self, dom_embedding_size=128,
            word_embedding_size=128, word_dict=["null"], dom_object_type_list=["input", "click", "text"],
            word_max_length=15, dom_max_length=15,
            value_function_layers=[64,32,1], value_activations=['tanh', 'tanh', 'sigmoid'],
            policy_function_layers=[64,32,3], policy_activations=['tanh', 'tanh', 'sigmoid'],
            word_vectors=np.zeros([1, 128]), name="BetaDOMNet"):
        self.name = name

        ################################ LAYERS for the model #####################################
        # self.dom_embedding
        self.dom_embedding_size = dom_embedding_size
        self.dom_embedding_vocab = Vocabulary(dom_object_type_list)
        self.dom_embedding_layer = tf.keras.layers.Embedding(self.dom_embedding_vocab.length,
                output_dim=self.dom_embedding_size,
                name="tag_embedder")

        # word embedding
        self.word_embedding_size = word_embedding_size
        self.word_dict = Vocabulary(word_dict)
        self.word_embedding_layer = tf.keras.layers.Embedding(self.word_dict.length,
                output_dim=self.word_embedding_size,
                embeddings_initializer=tf.keras.initializers.Constant(word_vectors),
                trainable=False, name="word_embedding_layer")

        self.word_bilstm_layer = tf.keras.layers.Bidirectional(layer=tf.keras.layers.LSTM(64,
                return_sequences=True, return_state=True),
                merge_mode='concat')

        # inputs for the model
        # dom_word_input: word label for the dom element
        self.dom_word_input = tf.keras.layers.Input(shape=[dom_max_length], dtype=tf.int32, name="dom_word_input")
        # dom_shape_input: bounding box
        self.dom_shape_input = tf.keras.layers.Input(shape=[dom_max_length, 4], dtype=tf.float32, name="bounding_box_input")
        self.dom_embedding_input = tf.keras.layers.Input(shape=[dom_max_length], dtype=tf.int32, name="dom_tag_input")
        # instruction_word_input:
        self.instruction_word_input = tf.keras.layers.Input(shape=[word_max_length], dtype=tf.int32, name="instruction_words")

        ################################ Bringing together the model #####################################
        dom_word_embedding = (self.word_embedding_layer(self.dom_word_input))
        dom_tag_embedding = (self.dom_embedding_layer(self.dom_embedding_input))
        dom_tag_embedding = (dom_tag_embedding)
        # shapecheck: dom_word_embedding: [batch_size, dom_max_length, word_embedding]
        dom_embedding = tf.concat([dom_word_embedding, self.dom_shape_input, dom_tag_embedding], axis=-1, name="full_dom_embedding")

        # instruction embedding
        instruction_embeddings = self.word_embedding_layer(self.instruction_word_input)
        lstm_results = self.word_bilstm_layer(instruction_embeddings)
        instruction_embedding, _ = self.attention_layer(64, lstm_results[1], lstm_results[0])

        # creating the final state
        state,_ = self.attention_layer(64, instruction_embedding, dom_embedding)

        # value function
        value = state
        for i,width in enumerate(value_function_layers):
            value = tf.keras.layers.Dense(width, activation=value_activations[i])(value)

        policy = state
        for i,width in enumerate(policy_function_layers):
            policy = tf.keras.layers.Dense(width, activation=policy_activations[i])(policy)

        model = tf.keras.Model(inputs=[self.dom_word_input, self.dom_shape_input, self.dom_embedding_input, self.instruction_word_input],
                outputs=[value, policy])
        tf.keras.utils.plot_model(model, 'model.png', show_shapes=True)

if __name__ == "__main__":
    b = BetaDOMNet()
