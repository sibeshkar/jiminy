# Jiminy

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

### Recording Demonstrations: 

Start the environment `wob.mini.ClickTest2-v0`in a container. This exposes a port with rewarder and VNC proxy. 

```
docker run -itd -e TURK_DB='' -p 5899:5899 --privileged --ipc host --cap-add SYS_ADMIN sibeshkar/jiminywob demonstration -e wob.mini.ClickTest2-v0
```

Connect to the environment by connecting any VNC client to the port `localhost:5899`. All recorded demonstrations are stored inside `/tmp/completed-demos`