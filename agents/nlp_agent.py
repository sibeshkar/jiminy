import argparse
import json
from jiminy.representation.structure import betaDOM
from jiminy.envs import SeleniumWoBEnv
from beta_dom_net import BetaDOMNet
import tensorflow as tf

parser = argparse.ArgumentParser("Language based agent for performing WoB tasks")
parser.add_argument('--remotes', dest='remotes',
        action='store', default='url_list.txt',
        help='File which contains URLs to the tasks we want to solve')
parser.add_argument('--config', dest='config',
         action='store', default='domnet_config.json',
         help='BetaDOMNet network configuration')
args = parser.parse_args()

if __name__ == "__main__":
    wobenv = SeleniumWoBEnv()
    with open(args.remotes, mode='r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    wobenv.configure(_n=len(lines), remotes=lines)
    betadom = betaDOM(wobenv)
    with open(args.config) as f:
        json_str = f.read()
    config = json.loads(json_str)
    word2vec_embeddings, word_list = load_embeddings(config['wv_model'])
    dom_object_list = ["input", "click", "text"]
    domnet = BetaDOMNet(dom_embedding_size=config["dom_embedding_size"],
        word_embedding_size=config["word_embedding_size"],
        word_dict=word_list, dom_object_type_list=dom_object_list,
        word_max_length=config["word_max_length"], dom_max_length=config["dom_max_length"],
        value_function_layers=config["value_function_layers"],
        value_activations=config["value_activations"],
        policy_function_layers=config["policy_function_layers"],
        policy_activations=config["policy_activations"],
        word_vectors=word2vec_embeddings,
        name=config["name"])
