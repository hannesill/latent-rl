from gymnasium import RewardWrapper
import numpy as np


class ConstantRewardWrapper(RewardWrapper):
    def __init__(self, env, constant_reward=1, std=0.1, seed=0):
        super().__init__(env)
        self.std = std
        self.constant_reward = constant_reward
        np.random.seed(seed)

    def reward(self, reward):
        noise = np.random.normal(0, self.std)
        return self.constant_reward + noise
