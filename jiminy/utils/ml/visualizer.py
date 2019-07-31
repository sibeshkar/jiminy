"""
Responsible for handling visualization of screens
during training of model to understand how well it works
"""
import cv2
import tensorflow as tf

colors = {
        "text" : (255, 0, 0),
        "input" : (0, 255, 0),
        "checkbox" : (0, 0, 255),
        "click" : (0, 127, 127)
}

class ScreenVisualizerCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, vocab, logdir, prefix, baseModel=None):
        super(ScreenVisualizerCallback, self).__init__()
        self.dataset = dataset
        self.vocab = vocab
        self.baseModel = baseModel

        self.logdir, self.prefix = logdir, prefix

    def on_epoch_end(self, epoch, logs=None):
        for i, img in enumerate(self.dataset):
            label, bounding_box = self.baseModel.forward_pass(img)
            screenVisualizer(img, bounding_box, label, epoch, self.logdir, self.prefix+"-{}".format(i))

def screenVisualizer(img, bounding_box, label, epoch, logdir, prefix):
    for i, bb in enumerate(bounding_box):
        color = (255, 0, 0)
        if label[i] in colors:
            color = colors[label[i]]
        cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), color, 1)
    path = "{}/{}-epoch-{}.png".format(logdir, prefix, epoch)
    assert cv2.imwrite(path, img), "Can not write to path: {}".format(path)

def getVisualizationList(dataset, num=10):
    for ((img, _), _) in dataset.take(1):
        img = img.numpy()
        return img
