import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


class LatentContinuousActionWrapper(gym.ActionWrapper):
    """
    This wrapper is used to project the latent action space to the original action space.
    It uses a non-linear transformation to do that.
    The default initialization of the non-linear transformation is:
        - orthogonal
        - sqrt(2) std,
        - activation tanh, except for the last layer which is linear
    """

    def __init__(self, env: gym.Env, latent_dim: int):
        super().__init__(env)
        self.latent_dim = latent_dim
        self.action_dim = env.action_space.shape[0]

        if self.latent_dim == 0:
            self.latent_dim = self.action_dim

        # Store original action space bounds
        self.original_low = self.env.action_space.low
        self.original_high = self.env.action_space.high

        # Specify new (latent) action space
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(latent_dim,), dtype=np.float32
        )

        # Initialize the projector
        latent_original_dim_diff = self.action_dim - self.latent_dim
        hidden_1_dim = self.latent_dim + int(0.33 * latent_original_dim_diff)
        hidden_2_dim = self.latent_dim + int(0.67 * latent_original_dim_diff)
        self.projector = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_1_dim),
            nn.ReLU(),
            nn.Linear(hidden_1_dim, hidden_2_dim),
            nn.ReLU(),
            nn.Linear(hidden_2_dim, self.action_dim),
            nn.Tanh() # output to [-1, 1]
        )

        # Initialize the weights with orthogonal initialization and sqrt(2) std
        for layer in self.projector:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, np.sqrt(2))
                nn.init.constant_(layer.bias, 0)

    def action(self, action):
        with ((torch.no_grad())):
            action = torch.tensor(action, dtype=torch.float32)
            action = self.projector(action)

            # Detach and convert to NumPy here
            action = action.detach().cpu().numpy()

            # Scale to the action space bounds (assuming symmetrical bounds)
            action = action * self.original_high
            action = np.clip(action, self.original_low, self.original_high)  # Additional safety

        return action

    def get_projector_weights(self):
        return self.projector.state_dict()

    def set_projector_weights(self, weights):
        self.projector.load_state_dict(weights)

