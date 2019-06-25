import tensorflow as tf
import numpy as np
import string

def attention_layer(units, query, values):
    W1 = tf.keras.layers.Dense(units)
    W2 = tf.keras.layers.Dense(units)
    V = tf.keras.layers.Dense(1)
    time_axis = tf.expand_dims(query, 1)
    score = V(tf.nn.tanh(W1(time_axis)
        + W2(values)))
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = tf.reduce_sum(values*attention_weights, axis=1)
    return context_vector, attention_weights

def process_text(text):
    words = text.split(" ")
    words = [word.strip().lower() for word in words]
    words = [''.join([i for i in filter(lambda ch : not ch in string.punctuation, word)]) for word in words]
    return words

def pad(vector, length, axis=0):
    pad_width = [(0,0) for _ in range(len(vector.shape))]
    pad_width[axis] = (0, length)
    return np.pad(vector, pad_width, 'constant')

def load_lines_ff(fname):
    with open(fname, mode="r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines

def get_action_probability_pair(x,y,bm,probs):
    action_log_prob = np.log(tf.squeeze(probs[0])[x]) + \
        np.log(tf.squeeze(probs[1])[y]) + \
        np.log(tf.squeeze(probs[2])[bm])
    return vnc_event.PointerEvent(x+1, y+1, bm), action_log_prob

