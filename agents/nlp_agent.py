import argparse
import json
from replay_buffer import ReplayBuffer
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

    # TODO(prannayk): create a classmethod to which config can be passed
    domnet = BetaDOMNet.from_config(config, betadom)

    ## implements A2C algorithm ##
