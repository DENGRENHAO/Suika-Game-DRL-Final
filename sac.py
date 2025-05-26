from suika_gym import SuikaEnv
from wrappers import CoordSizeToImage
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, SubprocVecEnv
from multiprocessing import freeze_support
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback
import wandb
import torch
import torch.nn as nn
import numpy as np
import datetime
from collections import deque
from feature_extractor import MyCombinedExtractor
from wandb_callback import WandbLoggingCallback

# import cv2

config = {
    "env_name": "suika-game-l1-v0",
    "policy_type": "MultiInputPolicy",
    "total_timesteps": 800000,
    "buffer_size": 50000,
    "batch_size": 128,
    "learning_rate": 3e-4,
    "learning_starts": 10000,
    "log_entries": [
        "train/actor_loss",
        "train/critic_loss",
        "train/ent_coef",
    ],
}
id = datetime.datetime.now().strftime("%m-%d_%H-%M")
run = wandb.init(
    project="suika-sb3-sac",
    name=id,
    config=config,
    settings=wandb.Settings(x_disable_stats=True),
)


def make_env():
    NUM_ENVS = 4
    def _init():
        env = gym.make(config["env_name"], render_mode="rgb_array")
        env = CoordSizeToImage(env=env)
        env = Monitor(env)
        return env
    return SubprocVecEnv([_init for _ in range(NUM_ENVS)])

def make_single_env():
    def _init():
        env = gym.make(config["env_name"], render_mode="rgb_array")
        env = CoordSizeToImage(env)
        env = Monitor(env)
        return env
    return DummyVecEnv([_init])

if __name__ == "__main__":
    freeze_support()
    train_env = make_single_env()
    eval_env = make_single_env()

    policy_kwargs = dict(
        features_extractor_class=MyCombinedExtractor,
    )

model = SAC(
    config["policy_type"],
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    buffer_size=config["buffer_size"],
    batch_size=config["batch_size"],
    learning_rate=config["learning_rate"],
    learning_starts=config["learning_starts"],
)
model.learn(
    total_timesteps=config["total_timesteps"],
    log_interval=10,  # episode
    progress_bar=True,
    callback=WandbLoggingCallback(
        eval_env=make_env(),
        save_dir=f"weights/sb3_sac/{id}",
        log_interval=500,
        log_entries=config["log_entries"],
        verbose=1,
    ),
)
run.finish()
