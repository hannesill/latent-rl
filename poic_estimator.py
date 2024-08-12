import optuna
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import logging


class PolicyNetworkContinuousAction(nn.Module):
    def __init__(self, env, act_fn="tanh", hidden_dim=64):
        super(PolicyNetworkContinuousAction, self).__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.action_dim)

        self.init_weights()

        activation_fn_d = {
            "tanh": torch.tanh,
            "linear": lambda x: x,
            "relu": lambda x: torch.relu,
        }
        assert (act_fn in activation_fn_d.keys())
        self.act_fn = activation_fn_d[act_fn]

        # Scale the output with the output function
        self.action_scale = env.action_space.high.max()
        self.output_fn = self.scale_continuous_action

    def scale_continuous_action(self, x):
        x = x.detach().numpy()
        return self.action_scale * np.tanh(x)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        x = self.act_fn(x)
        x = self.output_layer(x)
        return x

    def act(self, state):
        x = self.forward(state)
        return self.output_fn(x)

    def init_weights(self):
        """
        Initialize the weights of the neural network with a normal distribution
        :return: None
        """
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.constant_(self.output_layer.bias, 0.0)


class POICEstimator:
    def __init__(self, env, seed, latent_dim, num_samples, num_episodes):
        self.env = env
        self.seed = seed
        self.latent_dim = latent_dim
        self.num_samples = num_samples
        self.num_episodes = num_episodes

        # Initialize the env with a random projector
        self.policy_net = PolicyNetworkContinuousAction(env)

    def estimate(self):
        # Sample reward sequences for the different policies
        all_scores_per_param = []
        for _ in tqdm(range(self.num_samples)):
            # Generate parameter theta_i ~ p(theta) and set it to the policy
            # p(theta) is a normal distribution with mean 0 and variance 1
            self.policy_net.init_weights()

            # For each episode collect the cumulative reward (score)
            returns_episodes = []
            for _ in range(self.num_episodes):
                # Initialize the environment
                obs, _ = self.env.reset()
                episodic_return = 0
                steps = 0
                terminated, truncated = False, False
                while not terminated and not truncated:
                    action = self.policy_net.act(torch.tensor(obs, dtype=torch.float32))
                    obs, reward, terminated, truncated, _ = self.env.step(action)
                    episodic_return += reward
                    steps += 1
                returns_episodes.append(episodic_return)
            returns_episodes = np.array(returns_episodes)
            all_scores_per_param.append(returns_episodes)

        all_scores_per_param = np.array(all_scores_per_param)

        all_scores = all_scores_per_param.flatten()
        r_max = all_scores.max()

        # Optimize temperature for POIC
        def objective(trial):
            temperature = trial.suggest_loguniform('temperature', 1e-4, 2e4)
            p_o1 = np.exp((all_scores - r_max) / temperature).mean()
            p_o1_ts = np.exp((all_scores_per_param - r_max) / temperature).mean(axis=1)
            marginal = -p_o1 * np.log(p_o1 + 1e-12) - (1 - p_o1) * np.log(1 - p_o1 + 1e-12)
            conditional = np.mean(p_o1_ts * np.log(p_o1_ts + 1e-12) + (1 - p_o1_ts) * np.log(1 - p_o1_ts + 1e-12))
            mutual_information = marginal + conditional

            return mutual_information

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=200)

        # Log the best trial
        trial = study.best_trial

        return trial.value

