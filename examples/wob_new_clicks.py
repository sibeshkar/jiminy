import jiminy
import jiminy.gym as gym
import time
from lib import wob_vnc
from PIL import Image

if __name__ == "__main__":
    env = gym.make("VNC.Core-v0")
    env = jiminy.wrappers.experimental.SoftmaxClickMouse(env)
    env = wob_vnc.MiniWoBCropper(env)

    env.configure(env='sibeshkar/wob-v0', task='ClickButton', remotes='vnc://localhost:5902+15901')
    obs = env.reset()
    
    time.sleep(3) #TODO: This needs to be unnecessary

    while True:
        a = env.action_space.sample()
        obs, reward, is_done, info = env.step([a])
        if obs[0] is None:
            print("Env is still resetting...")
            continue
        break

    for idx in range(5000):
        time.sleep(0.1)
        a = env.action_space.sample()
        obs, reward, is_done, info = env.step([a])
        if obs[0] is None:
            print("Env is resetting...")
            continue
        print("Sampled action: ", a)
        print("Response are of index:", idx)
        print("Observation", obs[0])
        print("Reward", reward)
        print("Is done", is_done)
        print("Info", info)
        env.render()


        #im = Image.fromarray(obs[0])
        #im.save("test_frames/frame-%03d.png" % idx)

    env.close()
    pass
