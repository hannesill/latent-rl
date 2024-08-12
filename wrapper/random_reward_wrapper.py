from gymnasium import RewardWrapper
import numpy as np


class RandomRewardWrapper(RewardWrapper):
    def __init__(self, env, min_reward=-1, max_reward=1, seed=0):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
        np.random.seed(seed)

    def reward(self, reward):
        return np.random.uniform(self.min_reward, self.max_reward)
