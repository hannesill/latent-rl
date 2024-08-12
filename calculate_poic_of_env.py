import argparse

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from wrapper.constant_reward_wrapper import ConstantRewardWrapper
from wrapper.latent_wrapper import LatentContinuousActionWrapper
from wrapper.random_reward_wrapper import RandomRewardWrapper
from poic_estimator import POICEstimator
import torch


gamma = 0.99


def make_env(env_id, seed, cleanrl_setup, constant_reward, random_reward, latent_space):
    env = gym.make(env_id)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    env.reset(seed=seed)

    if cleanrl_setup:
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    if constant_reward:
        env = ConstantRewardWrapper(env, constant_reward=1, std=0, seed=seed)
    if random_reward:
        env = RandomRewardWrapper(env, min_reward=-1, max_reward=1, seed=seed)
    # Add a latent action space if specified
    if latent_space > 0:
        env = LatentContinuousActionWrapper(env, latent_dim=latent_space)
        # env.set_projector_weights(torch.load(f"projectors/worst_projector_HalfCheetah-v4_2024-08-08_10-55-06_s50_e30_l4.pt"))
    return env


def set_global_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    seed = 0

    # Get the command line arguments
    parser = argparse.ArgumentParser()
    # Environment specific arguments
    parser.add_argument("--env", type=str, default="Ant-v4", help="Open AI gym environments")
    parser.add_argument("--latent", type=int, default=0, help="Latent dimension")
    parser.add_argument("--constant_reward", action="store_true", help="Use constant reward")
    parser.add_argument("--random_reward", action="store_true", help="Use random reward")
    parser.add_argument("--cleanrl_setup", action="store_true", help="Use CleanRL's env setup")
    # POIC specific arguments
    parser.add_argument("--num_seeds", type=int, default=5, help="Number of seeds to try")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes")
    args = parser.parse_args()

    # Initialize a new env with a random projector
    env = make_env(args.env, seed, args.cleanrl_setup, args.constant_reward, args.random_reward, args.latent)

    # Initialize the hyperparameters
    num_samples = args.num_samples
    num_episodes = args.num_episodes
    num_seeds = args.num_seeds

    # Try out different seeds for higher confidence
    poic_scores = []
    for seed in tqdm(range(num_seeds)):
        # Set random seeds for reproducibility
        set_global_seeds(seed)

        # Initialize the POICEstimator with the new env
        poic_estimator = POICEstimator(env, seed, args.latent, num_samples, num_episodes)

        # Estimate the POIC
        poic = poic_estimator.estimate()

        poic_scores.append(poic)

    # Report the settings
    print(f"Settings: env={args.env}, cleanrl_setup={args.cleanrl_setup}, constant_reward={args.constant_reward}, random_reward={args.random_reward}, latent={args.latent}, num_seeds={args.num_seeds}, num_samples={args.num_samples}, num_episodes={args.num_episodes}")
    # Report the results with a precision of 5 dec
    print(f"Highest score: {round(max(poic_scores), 5)}")
    print(f"Lowest score: {round(min(poic_scores), 5)}")
    print(f"Average score: {round(sum(poic_scores) / len(poic_scores), 5)}")
