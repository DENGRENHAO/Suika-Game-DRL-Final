from suika_gym import SuikaEnv
from wrappers import CoordSizeToImage
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecVideoRecorder,
    SubprocVecEnv,
)
from stable_baselines3.common.callbacks import BaseCallback

import wandb
import torch
import torch.nn as nn
import numpy as np
import datetime
from collections import deque
from feature_extractor import MyCombinedExtractor

# import cv2

config = {
    "env_name": "suika-game-l1-v0",
    "policy_type": "MultiInputPolicy",
    "total_timesteps": 300000,
    "buffer_size": 500000,
    "batch_size": 256,
    "learning_rate": 2e-4,
}


class WandbLoggingCallback(BaseCallback):
    def __init__(self, eval_env, log_interval=500, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.interval = log_interval
        self.eval_interval = log_interval * 10
        self.ep_reward = 0
        self.rewards = deque(maxlen=100)
        self.scores = deque(maxlen=100)
        self.prev_log_timesteps = 0
        self.prev_eval_timesteps = 0

    def _on_step(self) -> bool:
        def evaluate():
            env = self.eval_env
            obs, info = env.reset()
            done = False
            frames = []
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                frames += env.unwrapped.render_states(states=info["fruit_states"])
            return info["score"], frames

        self.ep_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.rewards.append(self.ep_reward)
            self.scores.append(self.locals["infos"][0]["score"])
            self.ep_reward = 0

        if self.num_timesteps - self.prev_log_timesteps > self.interval:
            logs = {
                "train_reward": np.mean(self.rewards) if len(self.rewards) > 0 else 0,
                "train_scores": np.mean(self.scores) if len(self.scores) > 0 else 0,
                "actor_loss": self.logger.name_to_value["train/actor_loss"],
                "critic_loss": self.logger.name_to_value["train/critic_loss"],
                "ent_coef": self.logger.name_to_value["train/ent_coef"],
            }
            self.prev_log_timesteps = self.num_timesteps
            if self.num_timesteps - self.prev_eval_timesteps > self.eval_interval:
                score, frames = evaluate()
                print(f"Step {self.num_timesteps} eval score: {score:.2f}")
                frames = np.array(frames)
                frames = frames.transpose(0, 3, 1, 2)  # Convert to (T, C, H, W) formath
                logs["eval_score"] = score
                logs["video"] = wandb.Video(
                    frames,
                    fps=30,
                    format="mp4",
                )
                self.prev_eval_timesteps = self.num_timesteps
            wandb.log(logs, step=self.num_timesteps)
        return True


def make_env():
    env = gym.make(config["env_name"], render_mode="rgb_array")
    env = CoordSizeToImage(env=env)
    return env


if __name__ == "__main__":
    # env = make_env()
    vec_env = SubprocVecEnv([lambda: Monitor(make_env()) for _ in range(10)])

    policy_kwargs = dict(
        features_extractor_class=MyCombinedExtractor,
    )

    model = SAC(
        config["policy_type"],
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
    )

    run = wandb.init(
        project="suika-sb3-sac",
        name=datetime.datetime.now().strftime("%m-%d_%H-%M"),
        config=config,
        settings=wandb.Settings(x_disable_stats=True),
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        log_interval=10,  # episode
        progress_bar=True,
        callback=WandbLoggingCallback(make_env()),
    )
    # model.save("sac_model")
    run.finish()
