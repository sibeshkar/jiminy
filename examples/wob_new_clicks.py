import jiminy
import jiminy.gym as gym
import time
from PIL import Image

if __name__ == "__main__":
    env = gym.make("VNC.Core-v0")
    env = jiminy.wrappers.experimental.SoftmaxClickMouse(env)

    #env.configure(remotes='vnc://gpu:5900+15900')
    env.configure(env='sibeshkar/wob-v0', task='TicTacToe', remotes='vnc://localhost:5901+15901')
    obs = env.reset()

    while True:
        a = env.action_space.sample()
        obs, reward, is_done, info = env.step([a])
        if obs[0] is None:
            print("Env is still resetting...")
            continue
        break

    for idx in range(500):
        time.sleep(0.1)
        a = env.action_space.sample()
        obs, reward, is_done, info = env.step([a])
        if obs[0] is None:
            print("Env is resetting...")

            continue
        print("Sampled action: ", a)
        print("Response are of index:", idx)
        print("Observation", obs[0].shape)
        print("Reward", reward)
        print("Is done", is_done)
        print("Info", info)


        #im = Image.fromarray(obs[0])
        #im.save("test_frames/frame-%03d.png" % idx)

    env.close()
    pass