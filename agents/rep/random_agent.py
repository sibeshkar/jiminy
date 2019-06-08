from jiminy.envs import SeleniumWoBEnv as WobEnv
from jiminy.representation.structure import betaDOM

class RandomAgentWoB(object):
    def __init__(self, betadom):
        self.betadom  = betadom

    def sample_episode(self):
        self.betadom.env.reset()
        done = False
        reward_total = 0.
        while not done:
            for i in range(10):
                action_list = [self.betadom.env.action_space.sample() for _ in range(self.betadom.n)]
                obs, reward, done, info = self.betadom.env.step(action_list)
                reward_total += sum(reward)
            self.betadom.observation(obs)

        print(reward_total)

def set_bm(action_list, bm):
    for i in range(len(action_list)):
        action_list[i].buttonmask = bm
    return action_list

class RandomClickAgentWoB(object):
    def __init__(self, betadom):
        self.betadom  = betadom

    def sample_episode(self):
        self.betadom.env.reset()
        done = False
        reward_total = 0.
        while not done:
            for i in range(10):
                if done:
                    break
                action_list = [self.betadom.env.action_space.sample() for _ in range(self.betadom.n)]
                action_list = set_bm(action_list, 1)
                obs, reward, done, info = self.betadom.env.step(action_list)
                action_list = set_bm(action_list, 0)
                obs, reward, done, info = self.betadom.env.step(action_list)
                reward_total += sum(reward)
            self.betadom.observation(obs)
            print(self.betadom)

        print(reward_total)

if __name__ == "__main__":
    wobenv = WobEnv()
    wobenv.configure(_n=1, remotes=["file:///Users/prannayk/ongoing_projects/jiminy-project/miniwob-plusplus/html/miniwob/click-button.html"])
    betadom = betaDOM(wobenv)
    ra = RandomClickAgentWoB(betadom)
    for _ in range(10):
        ra.sample_episode()
