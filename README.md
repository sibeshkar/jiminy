# Jiminy - An imitation learning library

![Jiminy recording](utils/screencast2.gif)

##### (This repository is in pre-alpha stage. Expect a lot of errors.)


Jiminy is an imitation learning library that uses VNC as an interface, and is meant to train agents on any environment(starting with World-of-Bits) tasks.

### How to run sample:

#### 1. Run remote environment:

```

docker run -itd -p 5900:5900 -p 15900:15900 --ipc host --privileged --cap-add SYS_ADMIN sibeshkar/jiminywob

```

#### 2. Run Jiminy in docker container :

```

docker run -it --net host sibeshkar/jiminy

```

#### 3.Run sample random agent:

Above command should open a terminal to the container. Inside the container, run the sample agent like this:

```

cd examples && python wob_clicks.py

```

Wait a few moments for the remote environment to reset to the sample environment that the agent uses : `wob.mini.BisectAngle-v0`. Check `docker logs` of the remote environment container if this agent fails to connect. This agent interacts with the environment inside the remote container and returns a tuple of the form `(observation, reward, is_done, info)`with every interaction.

You can observe what the agent is doing while it runs, by connecting to the remote environment with any VNC client by pointing it towards `localhost:5900`

  
### Infrastructure Overview:
The core Jiminy infrastructure allows agents to train in parallel environments for asynchronous methods for RL (i.e A3C). This remote docker image starts a TigerVNC server and boots a Python control server, which uses Selenium to open a Chrome browser to an in-container page which loads [MiniWoB](https://stanfordnlp.github.io/miniwob-plusplus/) environments. The `utils` folder contains helpful bash files to handle the architecture. 

Follow these instructions to use:
1. Start 8 remote environments in docker containers - `./utils/start_docker.sh` 
2. [OPTIONAL] View the 8 environments, make sure [vncviewer](https://tigervnc.org/) is installed. - `./utils/vnc.sh` 
3. After completion, clean up using `./utils/clear_docker.sh`

Extras: Use `./utils/move.sh` to move vncviewers to a grid position, `./utils/kill.sh` to kill **only** the vncviewers. 

### Training an A3C agent from scratch:

Jiminy contains an example for training agents using A3C methods from scratch.  Follow these steps to reproduce: 

1. Clone and install this repository (preferably in virtualenv or conda). `pip install -e .`
2. Start 8 remote environments : `./utils/start_docker.sh`
3. [OPTIONAL] Open VNC viewer windows. `./utils/vnc.sh`
4. Move into the examples directory : `cd examples`
5. Install requirements for the agent: `pip install -r requirements.txt`
6. Train the agent : `./wob_click_train.py -n t1 --cuda` (t1 is the name of the iteration)

All runs are stored in the `./examples/runs/` directory , and best models are stored in `./examples/saves/`. You can inspect the training by starting `tensorboard --logdir=runs` in a separate terminal.

On a GTX 1050Ti, the above takes one hour, i.e. 100K-150K frames to get to a mean reward of 0.9. 



If you just want to see how Jiminy handles arrays of tuples of the form `(observation_n, reward_n, done_n, info)` from the parallel environments, just run ./wob	

### Recording Demonstrations:

Start the environment `wob.mini.ClickTest2-v0`in a container. This exposes a port with rewarder and VNC proxy.

```

docker run -itd -e TURK_DB='' -p 5899:5899 --privileged --ipc host --cap-add SYS_ADMIN sibeshkar/jiminywob demonstration -e wob.mini.ClickTest2-v0

```

Connect to the environment by connecting any VNC client to the port `localhost:5899`. All recorded demonstrations are stored inside `/tmp/completed-demos`

#### Coming soon:
1. How to record and play demonstrations
2. Using demonstrations during training