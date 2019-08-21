import jiminy
import jiminy.gym as gym
import time
import random

if __name__ == "__main__":
    env = gym.make('Robot-v0')
    env.configure(env='sibeshkar/wob-v1', task='ClickButton', remotes='http://0.0.0.0:15901')

    obs = env.reset()

    while True:
        a = env.action_space.sample()
        obs, reward, is_done, info = env.step([a])
        if obs[0] is None: #substitute with obs[0]['vision'] for image equivalent
            print("Env is still resetting...")
            continue
        break

    for idx in range(5000):
        time.sleep(random.uniform(0.1, 0.9))
        a = env.action_space.sample()
        obs, reward, is_done, info = env.step([a])
        if obs[0] is None:
            print("Env is resetting...")
            continue
        
        # if is_done[0]:
        #     print("Reward: {}, Done: {}, Info {}".format(reward, is_done, info['n'][0]['env_status.episode_id']))
        print("Sampled action: ", a)
        print("Response are of index:", idx)
        print("Observation", obs[0]['dom'])
        print("Reward", reward)
        print("Is done", is_done)
        print("Info", info)
        env.render()


        #im = Image.fromarray(obs[0])
        #im.save("test_frames/frame-%03d.png" % idx)

    env.close()
    pass