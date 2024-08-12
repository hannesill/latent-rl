import gymnasium as gym
from gymnasium import spaces
import numpy as np


class RandomRewardEnv(gym.Env):
    def __init__(self, action_dim=2, observation_dim=1):
        super(RandomRewardEnv, self).__init__()
        # Define the action and observation space
        self.action_space = spaces.Box(low=0, high=1, shape=(action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(observation_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        observation = np.array([0.5], dtype=np.float32)
        info = {}

        return observation, info

    def step(self, action):
        # The reward is a random number between 0 and 1
        reward = np.random.uniform(0, 1)
        terminated = False
        truncated = False
        info = {}
        # Randomly choose a new observation
        observation = np.array([np.random.uniform(0, 1)], dtype=np.float32)
        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass  # Implement rendering if necessary

    def close(self):
        pass