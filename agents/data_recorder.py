from jiminy.envs import SeleniumWoBEnv
import os
from jiminy.representation.structure import betaDOM
import argparse
import jiminy.utils as utils
from datetime import datetime
import json
import threading
import multiprocessing

arg_parser = argparse.ArgumentParser(description="Data recorder for Selenium to betaDOM")
arg_parser.add_argument('--logdir', dest='logdir', action='store',
        default='logdir', help="Choose the name of the logdir to be used with Data recorder")
arg_parser.add_argument('--task_list', dest='task_list', action='store',
        default='task_list.txt', help="Choose the file name containing the task_list to be used with Data recorder")
arg_parser.add_argument('--num_examples', dest='num_examples', action='store',
        default=100000, help="Choose the number of examples to record")

args = arg_parser.parse_args()

if __name__ == "__main__":
    utils.create_directory(args.logdir)
    os.putenv("JIMINY_LOGDIR", args.logdir + "/screenshots/")
    os.environ["JIMINY_LOGDIR"] = args.logdir + "/screenshots/" # TODO(prannayk) :  this is a hack -- why?

    assert os.path.exists(args.task_list), "Task list could not be loaded from {}".format(args.task_list)
    tasks = utils.get_lines_ff(args.task_list)
    cpu_count = multiprocessing.cpu_count()
    print("Using {} CPU cores".format(cpu_count))
    while len(tasks) < cpu_count:
        tasks += tasks

    jiminy_home = os.getenv("JIMINY_ROOT")
    prefix = "file:///{}/miniwob-plusplus/html/miniwob/".format(jiminy_home)
    tasks = ["{}{}.html".format(prefix,task) for task in tasks]
    n = len(tasks)

    wobenv = SeleniumWoBEnv()
    wobenv.configure(_n=n, remotes=tasks)
    betadom = betaDOM(wobenv)

    def env_runner(index):
        for _ in range(args.num_examples):
            obs = betadom.reset_runner(index)
            betadom.observation_runner(index, obs)
            jsonstring = str(betadom.betadom_instance_list[index])
            fname = args.logdir + "/" + datetime.now().strftime("%Y%m%d-%H%M%S%f") + ".json"
            with open(fname, mode="w") as f:
                json.dump(jsonstring, f)
            print("Written ... {}".format(fname))

    thread_list = []
    for i in range(n):
        t = threading.Thread(target=env_runner, args=(i,))
        t.start()
        thread_list.append(t)

    for thread in thread_list:
        thread.join()
