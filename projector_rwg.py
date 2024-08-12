import argparse
from datetime import datetime
import torch
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import logging

from poic_estimator import POICEstimator
from wrapper.latent_wrapper import LatentContinuousActionWrapper

logging.basicConfig(level=logging.INFO)

ROUNDING_PRECISION = 6

def set_global_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# TODO: Try out with CleanRL's env setup
def make_env(env_name, latent_dim, seed):
    env = gym.make(env_name)
    env = LatentContinuousActionWrapper(env, latent_dim)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    env.reset(seed=seed)
    return env


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--env", type=str, default="Ant-v4", help="Open AI gym environments")
    argparser.add_argument("--num_seeds", type=int, default=5, help="Number of random seeds to try")
    argparser.add_argument("--num_samples", type=int, default=100, help="Number of samples")
    argparser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes")
    argparser.add_argument("--num_projectors", type=int, default=10, help="Number of projectors to try")
    argparser.add_argument("--latent", type=int, default=0, help="Latent dimension")
    args = argparser.parse_args()

    # Initialize environment
    env_name = args.env
    latent_dim = args.latent

    # Initialize hyperparameters
    num_samples = args.num_samples
    num_episodes = args.num_episodes

    # Try out different projectors
    projectors_results = []
    for projector_run in tqdm(range(args.num_projectors)):
        # Initialize a new env with a random projector
        env = make_env(env_name, latent_dim, seed=np.random.randint(0, 1000))

        # Try out different seeds for higher confidence
        poic_scores = []
        for seed_run in tqdm(range(args.num_seeds)):
            # Set random seeds for reproducibility
            seed = np.random.randint(0, 1000)
            set_global_seeds(seed)

            # Initialize the POICEstimator with the new env
            poic_estimator = POICEstimator(env, seed, latent_dim, num_samples, num_episodes)

            # Estimate the POIC
            poic = poic_estimator.estimate()

            poic_scores.append(poic)

        # Save the projector
        projector = env.get_projector_weights()
        projectors_results.append({
            "projector_id": projector_run,
            "projector": projector,
            "poic_scores": poic_scores
        })

    # Log the results
    projectors_best_scores = []
    projectors_worst_scores = []
    projectors_average_scores = []
    all_projectors_scores = []
    for current_projector_id in range(args.num_projectors):
        current_projector_scores = projectors_results[current_projector_id]["poic_scores"]
        current_best_projector_score = max(current_projector_scores)
        current_worst_projector_score = min(current_projector_scores)
        current_average_projector_score = sum(current_projector_scores) / len(current_projector_scores)
        current_std_projector_score = np.std(current_projector_scores)

        projectors_best_scores.append(current_best_projector_score)
        projectors_worst_scores.append(current_worst_projector_score)
        projectors_average_scores.append(current_average_projector_score)
        all_projectors_scores.extend(current_projector_scores)

        logging.info(f"Projector {current_projector_id}: Best score: {round(current_best_projector_score, ROUNDING_PRECISION)}, Worst score: {round(current_worst_projector_score, ROUNDING_PRECISION)}, Average score: {round(current_average_projector_score, ROUNDING_PRECISION)}, Std: {round(current_std_projector_score, ROUNDING_PRECISION)}")

    # Overall metrics
    logging.info("--- Overall Metrics ---")
    best_projector_score = max(projectors_best_scores)
    best_scoring_projector_id = projectors_best_scores.index(max(projectors_best_scores))
    logging.info(f"Best POIC: {round(best_projector_score, ROUNDING_PRECISION)} (Projector {best_scoring_projector_id})")

    worst_projector_score = min(projectors_worst_scores)
    worst_scoring_projector_id = projectors_worst_scores.index(min(projectors_worst_scores))
    logging.info(f"Worst POIC: {round(worst_projector_score, ROUNDING_PRECISION)} (Projector {worst_scoring_projector_id})")

    logging.info(f"Average POIC: {round(sum(all_projectors_scores) / len(all_projectors_scores), ROUNDING_PRECISION)}")
    logging.info(f"Std of all POICs: {round(np.std(all_projectors_scores), ROUNDING_PRECISION)}")

    # Get the best and worst projector (based on their average POIC score)
    best_projector_id = projectors_average_scores.index(max(projectors_average_scores))
    worst_projector_id = projectors_average_scores.index(min(projectors_average_scores))

    # Get the current date and time and save the projector
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    torch.save(projectors_results[best_projector_id]["projector"], f"./projectors/best_projector_{env_name}_{current_datetime}_s{num_samples}_e{num_episodes}_l{latent_dim}.pt")
    torch.save(projectors_results[worst_projector_id]["projector"], f"./projectors/worst_projector_{env_name}_{current_datetime}_s{num_samples}_e{num_episodes}_l{latent_dim}.pt")