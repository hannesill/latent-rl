import argparse
import math

import optuna
import torch
import numpy as np
import gymnasium as gym
from torch import nn
from tqdm import tqdm
import logging

from wrapper.latent_wrapper import LatentContinuousActionWrapper

logging.basicConfig(level=logging.INFO)


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
            "relu": lambda x: np.maximum(0, x),
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
        nn.init.normal_(self.fc1.weight, mean=0.0, std=1)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1)
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=1)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.constant_(self.output_layer.bias, 0.0)


def set_global_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_env(env_name, latent_dim, seed):
    env = gym.make(env_name)
    if latent_dim > 0:
        env = LatentContinuousActionWrapper(env, latent_dim)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    env.reset(seed=seed)
    return env


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--env", type=str, default="Ant-v4", help="Open AI gym environments")
    argparser.add_argument("--seed", type=int, default=0, help="Random seed")
    argparser.add_argument("--num_samples", type=int, default=100, help="Number of samples")
    argparser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes")
    argparser.add_argument("--num_projectors", type=int, default=10, help="Number of projectors to try")
    argparser.add_argument("--latent", type=int, default=0, help="Latent dimension")
    args = argparser.parse_args()

    # Set random seeds for reproducibility
    seed = args.seed
    set_global_seeds(seed)

    # Initialize environment
    env_name = args.env
    latent_dim = args.latent
    env = make_env(env_name, latent_dim, seed)

    # Initialize hyperparameters
    num_samples = args.num_samples
    num_episodes = args.num_episodes

    # Try out different projectors
    projector_score_pairs = []
    for projector_run in tqdm(range(args.num_projectors)):
        # Initialize the env with a random projector

        # Initialize a random policy
        policy_net = PolicyNetworkContinuousAction(env)

        # Sample reward sequences for the different policies
        all_scores_per_param = []
        for samp_num in tqdm(range(num_samples)):
            # Generate parameter theta_i ~ p(theta) and set it to the policy
            # p(theta) is a normal distribution with mean 0 and variance 1
            policy_net.init_weights()

            # For each episode collect the cumulative reward (score)
            score_episodes = []
            for episode_num in range(num_episodes):

                # Initialize the environment
                obs, _ = env.reset()
                score = 0
                steps = 0
                terminated, truncated = False, False
                while not terminated and not truncated:
                    action = policy_net.act(torch.tensor(obs, dtype=torch.float32))
                    action = np.clip(action, -1, 1)  # Ensure action is within bounds
                    obs, reward, terminated, truncated, _ = env.step(action)
                    score += reward
                    steps += 1
                score_episodes.append(score)
            score_episodes = np.array(score_episodes)
            all_scores_per_param.append(score_episodes)

        all_scores_per_param = np.array(all_scores_per_param)
        all_mean_scores = all_scores_per_param.mean(axis=1)

        all_scores = all_scores_per_param.flatten()
        r_max = all_scores.max()
        r_min = all_mean_scores.min()
        r_mean = all_scores.mean()

        # Optimize temperature for POIC
        def objective(trial):
            temperature = trial.suggest_loguniform('temperature', 1e-4, 2e4)
            p_o1 = np.exp((all_scores-r_max)/temperature).mean()
            p_o1_ts = np.exp((all_scores_per_param-r_max)/temperature).mean(axis=1)
            marginal = -p_o1*np.log(p_o1 + 1e-12) - (1-p_o1)*np.log(1-p_o1 + 1e-12)
            conditional = np.mean(p_o1_ts*np.log(p_o1_ts + 1e-12) + (1-p_o1_ts)*np.log(1-p_o1_ts + 1e-12))
            mutual_information = marginal + conditional

            return mutual_information

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=200)

        # Log the best trial
        trial = study.best_trial
        temperature = trial.params['temperature']
        logging.info(f"Temperature: {temperature}")
        logging.info(f"Optimal mutual information: {trial.value}")

        # Save the projector
        if latent_dim > 0:
            projector = env.get_projector_weights()
            projector_score_pairs.append((projector, trial.value))

    # Report the results
    print(projector_score_pairs)
    # Best projector score
    best_projector_score = max(projector_score_pairs, key=lambda x: x[1])[1]
    logging.info(f"Best projector score: {best_projector_score}")

    # Worst projector score
    worst_projector_score = min(projector_score_pairs, key=lambda x: x[1])[1]
    logging.info(f"Worst projector score: {worst_projector_score}")

    # Average projector score
    projector_scores = [x[1] for x in projector_score_pairs]
    average_projector_score = sum(projector_scores) / len(projector_score_pairs)
    logging.info(f"Average projector score: {average_projector_score}")

    # Save the best projector
    best_projector = max(projector_score_pairs, key=lambda x: x[1])[0]
    torch.save(best_projector, f"./projectors/best_projector_{env_name}_{num_samples}_{num_episodes}_{latent_dim}.pt")
