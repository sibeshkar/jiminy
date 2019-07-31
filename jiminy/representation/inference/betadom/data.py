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
    img = cv2.imread(img_path)
    imgray = cv2.imread(img_path, 0)
    edges = cv2.Canny(img, 40, 200)
    img_bin, thresh = cv2.threshold(imgray, 100, 255, 0)
    if len(img_bin.shape) == 2:
        img_bin = np.expand_dims(img_bin, axis=-1)
    if len(edges.shape) == 2:
        edges = np.expand_dims(edges, axis=-1)
    img = np.concatenate([img, edges, img_bin], axis=-1)
    target_list = []
    for obj in data['base_object_list']:
        tag = np.array(vocab.to_sym([obj['objectType']]))
        boundingBox = np.array(list(obj['boundingBox'].values()))
        target_list.append(np.concatenate([tag, boundingBox]))
    target_list.append(np.concatenate([np.array(vocab.to_sym(["END"])),
        np.array([0, 0, 0, 0], dtype=np.int64)]))
    return img, target_list

def generate_targets(img, target_list, max_target_length=10, depth=10, screen_shape=None):
    img_list = []
    generated_target_list = []
    end_target_list = []
    targets = [0 for _ in range(max_target_length)]

    for target in target_list:
        img_list.append(img)
        t = np.array(targets, dtype=np.int64)
        generated_target_list.append(t)
        end_target_list.append(target)
        targets = targets[1:] + [target[0]]

    img_list = np.array(img_list).astype(np.float32)
    img_dataset = tf.convert_to_tensor(img_list)

    generated_target_dataset = tf.convert_to_tensor(generated_target_list, dtype=tf.int64)
    # generated_target_dataset = tf.one_hot(generated_target_dataset, on_value=1.0, off_value=0., depth=depth)

    end_target_list = np.array(end_target_list)
    print(end_target_list)

    target_dataset = tf.convert_to_tensor(end_target_list[:,0].astype(np.int64), dtype=tf.int64)
    target_dataset = tf.one_hot(target_dataset, depth=depth, on_value=1.0, off_value=0.)

    if screen_shape is None:
        target_bb_dataset = tf.convert_to_tensor(end_target_list[:,1:].astype(np.float32), dtype=tf.float32)
        return tf.tuple([img_dataset, generated_target_dataset, target_dataset, target_bb_dataset])

    target_bb_dataset = [tf.convert_to_tensor(end_target_list[:,i+1].astype(np.int64), dtype=tf.int64) for i in range(4)]
    target_bb_dataset = [tf.one_hot(target_bb_dataset, depth=screen_shape[i%2], on_value=1.0, off_value=0.) for i in range(4)]
    return tf.tuple([img_dataset, generated_target_dataset, target_dataset] + target_bb_dataset)

def transform_to_slices(vocab, max_target_length=10, screen_shape=None):
    def transform(fname):
        img, target_list = load_from_file(fname, vocab)
        return generate_targets(img, target_list, max_target_length=10, depth=vocab.length, screen_shape=screen_shape)
    return transform

def tf_py_function(fn, types, shapes):
    def thunked_fn(fname):
        tf_result = tf.py_function(inp=(fname,), func=fn, Tout=types)
        assert len(tf_result) == len(shapes), "Shape mismatch : {}, {}".format(tf_result, shapes)
        n = len(tf_result)

        tf_result = [tf.reshape(tf_result[i], (-1,) + shapes[i]) for i in range(n)]

        tf_result = [tf.data.Dataset.from_tensor_slices(tf_result[i]) for i in range(n)]

        input_dataset = tf.data.Dataset.zip(tuple(tf_result[:2]))
        output_dataset = tf.data.Dataset.zip(tuple(tf_result[2:][::-1]))
        return tf.data.Dataset.zip((input_dataset, output_dataset))
    return thunked_fn

def create_dataset(dir_name, batch_size, vocab, max_target_length, screen_shape, num_files=10000, one_hot=False):
    assert vocab.to_sym(["START"]) == [0], "Expected first value in Vocab to be START, got: {}".format(vocab.to_key([0])[0])
    dir_name = dataroot + dir_name
    assert os.path.exists(dir_name), "Path does not exist: {}".format(dir_name)

    files = tf.data.Dataset.list_files(dir_name + "/*.json", shuffle=True).take(num_files)
    if one_hot:
        dataset = files.flat_map(tf_py_function(fn=transform_to_slices(vocab, max_target_length, screen_shape),
            types=(tf.float32, tf.int64, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
            shapes=(screen_shape, (max_target_length,), (vocab.length,),
                (screen_shape[0],), (screen_shape[1],), (screen_shape[0],), (screen_shape[1],)
                )
            ))
    else:
        dataset = files.flat_map(tf_py_function(fn=transform_to_slices(vocab, max_target_length, screen_shape=None),
            types=(tf.float32, tf.int64, tf.float32, tf.float32),
            shapes=(screen_shape, (max_target_length,), (vocab.length,),
                (4,)
                )
            ))

    dataset = dataset.shuffle(batch_size // 4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1000*batch_size)
    return dataset


if __name__ == "__main__":
    vocab = Vocabulary(["text","input", "checkbox", "button", "click"])
    dataset = create_dataset("logdir", 32, vocab, 10, (300, 300,3))
    for e in dataset.take(4):
        print(e[1])
        print([v.shape for v in e])
