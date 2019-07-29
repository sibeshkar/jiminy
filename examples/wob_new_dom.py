import jiminy
import jiminy.gym as gym
import time
from lib import wob_vnc
from PIL import Image
from jiminy.global_protos import DomObjectInstance
from google.protobuf import json_format

if __name__ == "__main__":
    env = gym.make("VNC.Core-v0")
    env = jiminy.actions.experimental.SoftmaxClickMouse(env)

    env.configure(env='prannayk/wob-v1', task='ClickButton', remotes='vnc://localhost:5900+15900')
    obs = env.reset()

    while True:
        a = env.action_space.sample()
        obs, reward, is_done, info = env.step([a])
        if obs[0]['dom'] is None:
            print("Env is still resetting...")
            continue
        break

    for idx in range(5000):
        domobj = DomObjectInstance()
        time.sleep(0.05)
        a = env.action_space.sample()
        obs, reward, is_done, info = env.step([a])
        if obs[0] is None:
            print("Env is resetting...")
            continue
        print("Sampled action: ", a)
        print("Response are of index:", idx)
        print("Observation", json_format.Parse(obs[0]["dom"], domobj))
        # print("Observation", domobj.ParseFromString(bytes(obs[0]["dom"], 'utf-8')))
        print("Reward", reward)
        print("Is done", is_done)
        print("Info", info)
        env.render()


        #im = Image.fromarray(obs[0])
        #im.save("test_frames/frame-%03d.png" % idx)

    env.close()
    pass
