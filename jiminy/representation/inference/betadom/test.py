from jiminy.utils.ml import Vocabulary, getVisualizationList, screenVisualizer
import tensorflow as tf
tf.enable_eager_execution()
from jiminy.representation.inference.betadom import BaseModelTrainer
import datetime
import argparse
import os
import json

parser = argparse.ArgumentParser(description="BaseModelTrainer settings")
parser.add_argument("--model_name", dest="model_name", action="store",
        default="baseModel.h5", help="Model name to which training has to be stored")
parser.add_argument("--model_type", dest="model_type", action="store",
        default="basic", type=str, help="Model type to be used for training")
parser.add_argument("--model_config", dest="model_config", action="store",
        default="model_config/small.json", help="Defines the basic model which is smaller than stored params")
args = parser.parse_args()


if __name__ == "__main__":
    start_time = datetime.datetime.now().strftime("%d-%b-%Y::%H-%M")
    vocab = Vocabulary(["START", "text","input", "checkbox", "button", "click", "END"])

    config_dict = dict()
    if os.path.exists(args.model_config):
        with open(args.model_config) as f:
            json_obj = json.load(f)
        config_dict = dict(json_obj)
    bmt = BaseModelTrainer(vocab=vocab, model_type=args.model_type, config=config_dict)

    bmt.baseModel.model.load_weights(args.model_name)

    visualization_img_list = getVisualizationList(bmt.dataset)
    logdir = "./logdir"
    prefix = "{}".format(start_time)

    for i,img in enumerate(visualization_img_list):
        label, bb = bmt.baseModel.forward_pass(img)
        screenVisualizer(img, bb, label, 0, "logdir", "test-set"+"-{}".format(i))
