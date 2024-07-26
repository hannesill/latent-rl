import argparse
import math

import optuna
import torch
import numpy as np
import gymnasium as gym
from torch import nn
from tqdm import tqdm
import logging

from poic_estimator import POICEstimator
from wrapper.latent_wrapper import LatentContinuousActionWrapper

logging.basicConfig(level=logging.INFO)


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
        # Initialize a new env with a random projector
        env = make_env(env_name, latent_dim, seed)

        # Initialize the POICEstimator with the new env
        # TODO: Mehrere seed Werte verwenden um mehr Confidence zu erhalten
        poic_estimator = POICEstimator(env, seed, latent_dim, num_samples, num_episodes)

        # Estimate the POIC
        poic = poic_estimator.estimate()

        # Save the projector
        if latent_dim > 0:
            projector = env.get_projector_weights()
            projector_score_pairs.append((projector, poic))

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
