import jiminy
import jiminy.gym as gym
import time

from lib import wob_vnc

REMOTES_COUNT = 8

if __name__ == "__main__":
    env = gym.make("VNC.Core-v0")
    env = jiminy.wrappers.experimental.SoftmaxClickMouse(env)
    remotes_url= wob_vnc.remotes_url(port_ofs=0, hostname='0.0.0.0', count=REMOTES_COUNT)
    env.configure(env='sibeshkar/wob-v1', task='ClickButton', remotes=remotes_url)
    observation_n = env.reset()

    while True:
        action_n = [env.action_space.sample() for ob in observation_n]
        observation_n, reward_n, done_n, info = env.step(action_n)
        if observation_n[0]['dom'] is None: #substitute with obs[0]['vision'] for image equivalent
            print("Env is still resetting...")
            continue
        break
    idx = 0
    while True:
        time.sleep(0.05)
        #a = env.action_space.sample()
        action_n = [env.action_space.sample() for ob in observation_n]
        observation_n, reward_n, done_n, info = env.step(action_n)
        if observation_n[0]['dom'] is None:
            print("Env is resetting...")
            continue
        print("Sampled action: ", action_n)
        print("Response are of index:", idx)
        print("Observation", observation_n[0]['dom'])
        print("Reward", reward_n)
        print("Is done", done_n)
        print("Info", info)
        idx+=1
        env.render()


        #im = Image.fromarray(obs[0])
        #im.save("test_frames/frame-%03d.png" % idx)

    env.close()
    pass
