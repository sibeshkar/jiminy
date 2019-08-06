import jiminy
from jiminy.representation.structure import betaDOM
from jiminy import gym
from jiminy import vectorized
# from jiminy.spaces import vnc_event
# from jiminy.envs import SeleniumWoBEnv
from jiminy.utils.ml import Vocabulary
from jiminy.utils.lib import wob_vnc
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import utils
from a3c import A3C
import time
import sys
import logging
import os

# from file_initializer import FileInitializer

def convert_to_tf_dtype(dtype):
    if dtype == np.float32 or np.float64:
        return tf.float32
    if dtype == np.int32 or np.int64:
        return tf.int64
    return tf.float32

class BetaDOMNet(vectorized.Wrapper):
    def __init__(self, env=None, dom_embedding_size=128, dom_max_length=15,
            word_embedding_size=128, word_dict=["null"],
            dom_object_type_list=["input", "click", "text"], word_max_length=15,
            value_function_layers=[64,32,1], value_activations=['tanh', 'tanh', 'tanh'],
            policy_function_layers=[64,32,3], policy_activations=['tanh', 'tanh', 'sigmoid'],
            greedy_epsilon=1e-1, word_vectors=np.zeros([1, 128]), name="BetaDOMNet", offsets=(0,0)):
        assert (not env is None), "Env can not be null"
        self.env = env
        self.greedy_epsilon = greedy_epsilon
        self.name = name
        self.dom_max_length = dom_max_length
        self.word_max_length = word_max_length
        self.dom_embedding_size = dom_embedding_size
        self.word_embedding_size = word_embedding_size
        self.dom_object_type_list = dom_object_type_list
        self.word_dict = word_dict
        self.value_function_layers = value_function_layers
        self.value_activations = value_activations
        self.policy_function_layers = policy_function_layers
        self.policy_activations = policy_activations
        self.offsets = offsets

    def create_model(self):
        ################################ LAYERS for the model #####################################
        # self.dom_embedding
        self.dom_embedding_vocab = Vocabulary(self.dom_object_type_list)
        self.dom_embedding_layer = tf.keras.layers.Embedding(self.dom_embedding_vocab.length,
                output_dim=self.dom_embedding_size,
                name="tag_embedder")

        # word embedding
        self.word_dict = Vocabulary(self.word_dict)
        self.word_embedding_layer = tf.keras.layers.Embedding(self.word_dict.length,
                output_dim=self.word_embedding_size,
                # embeddings_initializer=tf.keras.initializers.Constant(word_vectors),
                trainable=False, name="word_embedding_layer")

        self.word_bilstm_layer = tf.keras.layers.Bidirectional(layer=tf.keras.layers.LSTM(64,
                return_sequences=True, return_state=True),
                merge_mode='concat')

        # inputs for the model
        # dom_word_input: word label for the dom element
        self.dom_word_input = tf.keras.Input(shape=[self.dom_max_length], dtype=tf.int32, name="dom_word_input")
        # dom_shape_input: bounding box
        self.dom_shape_input = tf.keras.Input(shape=[self.dom_max_length, 4], dtype=tf.float32, name="bounding_box_input")
        # dom_embedding_input: label for the dom element of the element -- driven by objectType in JiminyBaseObject
        self.dom_embedding_input = tf.keras.Input(shape=[self.dom_max_length], dtype=tf.int32, name="dom_tag_input")
        # instruction_word_input:
        self.instruction_word_input = tf.keras.Input(shape=[self.word_max_length], dtype=tf.int32, name="instruction_words")

        ################################ Bringing together the model #####################################
        dom_word_embedding = self.word_embedding_layer(self.dom_word_input)
        dom_tag_embedding = self.dom_embedding_layer(self.dom_embedding_input)
        # dom_tag_embedding = (dom_tag_embedding)
        # shapecheck: dom_word_embedding: [batch_size, dom_max_length, word_embedding]
        dom_embedding = tf.concat([dom_word_embedding, self.dom_shape_input, dom_tag_embedding], axis=-1, name="full_dom_embedding")

        # instruction embedding
        instruction_embeddings = self.word_embedding_layer(self.instruction_word_input)
        lstm_results = self.word_bilstm_layer(instruction_embeddings)
        instruction_embedding, _ = utils.attention_layer(64, lstm_results[1], lstm_results[0])

        # creating the final state
        state,_ = utils.attention_layer(64, instruction_embedding, dom_embedding)

        # value function
        value = state
        for i,width in enumerate(self.value_function_layers):
            value = tf.keras.layers.Dense(width, activation=self.value_activations[i])(value)


        policy = state
        for i,width in enumerate(self.policy_function_layers):
            policy = tf.keras.layers.Dense(width, activation=self.policy_activations[i])(policy)
        policy_x = tf.keras.layers.Dense(self.env.action_space.n, activation='softmax', name='x-coordinate-action')(policy)
        # policy_y = tf.keras.layers.Dense(self.env.screen_shape[1], activation='softmax', name='y-coordinate-action')(policy)
        # policy_b = tf.keras.layers.Dense(2, activation='softmax', name='button-mask-action')(policy)

        self.model = tf.keras.Model(inputs=[self.dom_word_input, self.dom_shape_input, self.dom_embedding_input, self.instruction_word_input],
                outputs=[value, [policy_x,]])
        self.model.run_eagerly = True
        tf.keras.utils.plot_model(self.model, 'model.png', show_shapes=True)
        self.model.summary()


    @classmethod
    def from_config(cls, config, betadom):
        return cls(dom_embedding_size=config["dom_embedding_size"],
                word_embedding_size=config["word_embedding_size"],
                word_dict=["previous", "submit", "next", "none", "click", "on", "the", "yes", "no"],
                dom_object_type_list=config["dom_object_list"],
                word_max_length=config["word_max_length"],
                dom_max_length=config["dom_max_length"],
                value_function_layers=config["value_function_layers"],
                value_activations=config["value_activations"],
                policy_function_layers=config["policy_function_layers"],
                policy_activations=config["policy_activations"],
                word_vectors=None, name=config["name"], betadom=betadom)

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        if os.path.exists(path):
            print("Loading model from: ", path)
            self.model.load_weights(path)

    def step_runner(self, index, obs, model=None):
        if model is None:
            model = self.model
        betadom_instance = obs
        model_input = self.step_instance_input(betadom_instance)
        if model_input is None:
            logging.debug("Returning cuz shit")
            return betadom_instance, None, None, None
        value, policy = model(model_input)
        action, action_log_prob = self.step_policy(policy)
        return betadom_instance, value[0], action, action_log_prob

    def step(self, obs, model=None):
        if model is None:
            model = self.model
        model_input_list = []
        for i in range(self.n):
            model_input = self.step_instance_input(obs[i])
            model_input_list.append(model_input)
        assert len(model_input_list) == self.n, "Expected model_input_list to be of size {} but got {} : {}".format(self.n, len(model_input_list), model_input_list)
        model_input = [np.concatenate([model_input_list[j][i] for j in range(self.n)],
            axis=0) for i in range(4)]
        value, policy = model(model_input)
        action_list = []
        action_log_prob_list = []
        for i in range(self.n):
            action, action_log_prob = self.step_policy([
                policy[0][i],
                ])
            action_list.append(action)
            action_log_prob_list.append(action_log_prob)
        return obs, value, action_list, action_log_prob_list

    def step_policy(self, policy_raw):
        # make this nicer??
        # action wrapper not done properly
        ru = np.random.uniform()
        sample = self.env.action_space.sample()
        if ru > self.greedy_epsilon:
            sample = np.argmax(tf.squeeze(policy_raw[0]))
        return sample, utils.get_action_probability(sample, policy_raw)

    def step_instance_input(self, betadom_instance):
        instruction = utils.process_text(betadom_instance.query)
        instruction_input = utils.pad(self.word_dict.to_sym(instruction),
                self.word_max_length, axis=0)
        logging.debug(betadom_instance.objects)
        clickable_object_list = [obj for obj in filter(lambda x: x.type == 'click',
            betadom_instance.objects)]
        if len(clickable_object_list) == 0:
            return None
        tag_input = utils.pad(
                self.dom_embedding_vocab.to_sym([obj.type
                    for obj in clickable_object_list]),
                self.dom_max_length, axis=0)
        tag_text_input = utils.pad(
                self.word_dict.to_sym([''.join(utils.process_text(obj.content))
                    for obj in clickable_object_list]),
                self.dom_max_length, axis=0)
        tag_bounding_box = utils.pad(
                np.array([np.array(utils.process_bounding_box(obj.boundingBox), dtype=np.float32)
                    for obj in clickable_object_list]),
                self.dom_max_length, axis=0)
        model_input = [tag_text_input, tag_bounding_box, tag_input, instruction_input]
        model_input = [tf.convert_to_tensor(np.expand_dims(arr, 0), convert_to_tf_dtype(arr.dtype)) for arr in model_input]
        # print([inp.shape.as_list() for inp in model_input])
        return model_input

    def get_runner_instance(self):
        return tf.keras.models.clone_model(self.model)

    def configure(self, *args, **kwargs):
        self.env.configure(*args, **kwargs)
        self.create_model()

    def reset(self):
        return self.env.reset()

    def setupEnv(self):
        obs = self.env.reset()
        waitTime = 200
        first_set = False
        for idx in range(900000 // waitTime):
            a = self.env.action_space.sample()
            obs, reward, is_done, info = self.env.step([a for _ in range(self.n)])
            if obs[0] is None:
                if not first_set:
                    print("Env is still resetting...", end="")
                    first_set = True
                elif idx % 10 == 0:
                    print(".", end=" ")
                else:
                    print(".", end="")
                sys.stdout.flush()
                time.sleep(1. / waitTime)
                continue
            print()
            return
        assert False, "Set up took too long"


if __name__ == "__main__":
    screen_shape = (160, 210)
    env = gym.make("VNC.Core-v0")
    env = jiminy.actions.experimental.SoftmaxClickMouse(env, discrete_mouse_step=10)
    env = betaDOM(env)
    env = BetaDOMNet(env, greedy_epsilon=1e-1, offsets=(0, 75))
    remotes_url= wob_vnc.remotes_url(port_ofs=0, hostname='localhost', count=4)
    env.configure(screen_shape=screen_shape, env='sibeshkar/wob-v1', task='ClickButton',
            remotes=remotes_url)
    env.setupEnv()
    a3c = A3C(learning_rate=1e-2)
    a3c.learn(env)
