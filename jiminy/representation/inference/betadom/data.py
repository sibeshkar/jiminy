"""
The format for data that is generated by data.py
is a pair consisting of a screenshot (256x256x3 -- RGB)
and a list of objects with information about their type and their
relevant bounding box

[   (np.array((300,300,3), dtype=np.float32),
    [
    (np.array((1,), np.int32), np.array((4,), dtype=np.float32))
    ])]

 >>>>>
We get examples of the form:

    (np.ndarray([300,300,3]), np.ndarray([max_length]), np.ndarray([5]))

"""
from jiminy.utils.ml import Vocabulary
import tensorflow as tf
tf.enable_eager_execution()
import os
import json
import cv2
import numpy as np

dataroot = os.getenv("JIMINY_DATAROOT")

def load_from_file(fname, vocab):
    """
    Loads data from a specific filename prefix
    and returns it in the required format
    """
    fname = fname.numpy().decode()
    assert not dataroot is None, "JIMINY_DATAROOT Environment variable must be set"

    data_path = fname
    assert os.path.exists(data_path), "Data Loader File does not exist: {}".format(fname)

    with open(data_path, mode='r') as f:
        data = json.load(f)
        data = json.loads(data)

    img_path = os.getenv("JIMINY_DATAROOT") + data["screenshot_img_path"]
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = np.flip(img, axis=-1)
    target_list = []
    for obj in data['base_object_list']:
        tag = np.array(vocab.to_sym([obj['objectType']]))
        boundingBox = np.array(list(obj['boundingBox'].values()))
        target_list.append(np.concatenate([tag, boundingBox]))
    return img, target_list

def generate_targets(img, target_list, max_target_length=10):
    img_list = []
    generated_target_list = []
    end_target_list = []
    targets = [0 for _ in range(max_target_length)]

    for target in target_list:
        img_list.append(img)
        t = np.array(targets, dtype=np.int64)
        generated_target_list.append(t)
        end_target_list.append(target)
        targets = targets[:(max_target_length-1)] + [target[0]]
    img_dataset = tf.convert_to_tensor(img_list)
    generated_target_dataset = tf.convert_to_tensor(generated_target_list, dtype=tf.int64)
    end_target_list = np.array(end_target_list)
    target_dataset = tf.convert_to_tensor(end_target_list.astype(np.int64), dtype=tf.int64)
    return tf.tuple([img_dataset, generated_target_dataset, target_dataset])

def transform_to_slices(vocab, max_target_length=10):
    def transform(fname):
        img, target_list = load_from_file(fname, vocab)
        return generate_targets(img, target_list, max_target_length=10)
    return transform

def tf_py_function(fn, types, shapes):
    def thunked_fn(fname):
        tf_result = tf.py_function(inp=(fname,), func=fn, Tout=types)
        assert len(tf_result) == len(shapes), "Shape mismatch : {}, {}".format(tf_result, shapes)
        n = len(tf_result)
        tf_result = [tf.data.Dataset.from_tensor_slices(tf_result[i]) for i in range(n)]
        return tf.data.Dataset.zip(tuple(tf_result))
    return thunked_fn

def create_dataset(dir_name, batch_size, vocab, max_target_length, screen_shape):
    dir_name = dataroot + dir_name
    assert os.path.exists(dir_name), "Path does not exist: {}".format(dir_name)

    files = tf.data.Dataset.list_files(dir_name + "/*.json", shuffle=True)
    dataset = files.flat_map(tf_py_function(fn=transform_to_slices(vocab, max_target_length),
        types=(tf.int32, tf.int64, tf.int64),
        shapes=(screen_shape, (max_target_length,), (5,))))
    dataset = dataset.shuffle(batch_size // 4)
    dataset = dataset.batch(batch_size)
    return dataset


if __name__ == "__main__":
    vocab = Vocabulary(["text","input", "checkbox", "button", "click"])
    dataset = create_dataset("logdir", 32, vocab, 10, (300, 300,3))
    for e in dataset.take(4):
        print([v.shape for v in e])
