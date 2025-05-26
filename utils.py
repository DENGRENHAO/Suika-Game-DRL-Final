from wrappers import CoordSizeToImage
import gymnasium as gym
import random
import numpy as np
import torch


def set_seeds(seed):
    # 1. Set seeds for Python, NumPy, and PyTorch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 2. Set PyTorch to deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_env(env_name, seed):
    env = gym.make(env_name, render_mode="rgb_array", seed=seed)
    env = CoordSizeToImage(env=env)
    return env
