import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


class LatentContinuousActionWrapper(gym.ActionWrapper):
    """
    This wrapper is used to project the latent action space to the original action space.
    It uses a linear transformation to do that.
    The default initialization of the linear transformation is:
        - orthogonal
        - sqrt(2) std,
        - activation tanh,
        - 1-2 layer each 2-10 times action space size
    """
    hidden_size_factor = 2

    def __init__(self, env: gym.Env, latent_dim: int):
        super().__init__(env)
        self.latent_dim = latent_dim
        self.action_dim = env.action_space.shape[0]

        # Store original action space bounds
        self.original_low = self.env.action_space.low
        self.original_high = self.env.action_space.high

        # Initialize the projector
        self.projector = nn.Sequential(
            nn.Linear(self.latent_dim, int(0.33 * self.action_dim)),
            nn.Tanh(),
            nn.Linear(int(0.33 * self.action_dim), int(0.67 * self.action_dim)),
            nn.Tanh(),
            nn.Linear(int(0.67 * self.action_dim), self.action_dim),
            # No final Tanh to avoid squashing to [-1, 1]
        )
        # Initialize the weights with orthogonal initialization and sqrt(2) std
        for layer in self.projector:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, np.sqrt(2))
                nn.init.constant_(layer.bias, 0)

        # Specify new action space
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(latent_dim,), dtype=np.float32
        )

    def action(self, action):
        action = torch.tensor(action, dtype=torch.float32)
        action = self.projector(action)

        # Detach and convert to NumPy here
        action = action.detach().cpu().numpy()

        # Scale and shift to match original action space
        action = action * (self.original_high - self.original_low) / 2.0 + (
                    self.original_high + self.original_low) / 2.0

        # Clip to original action space bounds using NumPy
        action = np.clip(action, self.original_low, self.original_high)

        return action

    def get_projector_weights(self):
        return self.projector.state_dict()

    def set_projector_weights(self, weights):
        self.projector.load_state_dict(weights)

