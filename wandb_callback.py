from suika_gym import SuikaEnv
from wrappers import CoordSizeToImage
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback
import wandb
import torch
import torch.nn as nn
import numpy as np
import datetime
from collections import deque
from feature_extractor import MyCombinedExtractor
from os import makedirs, path


class WandbLoggingCallback(BaseCallback):
    def __init__(self, eval_env, save_dir, log_interval=500, log_entries=[], verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.interval = log_interval
        self.eval_interval = log_interval * 10
        self.ep_reward = 0
        self.rewards = deque(maxlen=100)
        self.scores = deque(maxlen=100)
        self.prev_log_timesteps = 0
        self.prev_eval_timesteps = 0
        self.log_entries = log_entries
        self.save_dir = save_dir
        if not path.exists(self.save_dir):
            makedirs(self.save_dir)
        self.best_mean_score = -np.inf

    def _on_step(self) -> bool:
        def evaluate(n=10):
            def _evaluate():
                env = self.eval_env
                obs, info = env.reset()
                done = False
                frames = []
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(action)
                    frames += env.unwrapped.render_states(states=info["fruit_states"])
                return info["score"], frames

            scores = []
            best_score = -np.inf
            best_frames = []
            for _ in range(n):
                score, frames = _evaluate()
                scores.append(score)
                if score > best_score:
                    best_score = score
                    best_frames = frames

            return np.mean(scores), np.std(scores), best_score, best_frames

        self.ep_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.rewards.append(self.ep_reward)
            self.scores.append(self.locals["infos"][0]["score"])
            self.ep_reward = 0

        if self.num_timesteps - self.prev_log_timesteps >= self.interval:
            logs = {
                "train/mean_reward": (
                    np.mean(self.rewards) if len(self.rewards) > 0 else 0
                ),
                "train/mean_scores": (
                    np.mean(self.scores) if len(self.scores) > 0 else 0
                ),
            }
            for entry in self.log_entries:
                if entry in self.logger.name_to_value:
                    logs[entry] = self.logger.name_to_value[entry]
            if self.num_timesteps - self.prev_eval_timesteps >= self.eval_interval:
                mean_score, std_score, best_score, frames = evaluate()
                print(f"Step {self.num_timesteps} eval score: {mean_score:.2f}")
                frames = np.array(frames)
                frames = frames.transpose(0, 3, 1, 2)  # Convert to (T, C, H, W) format
                logs["eval/mean_score"] = mean_score
                logs["eval/std_score"] = std_score
                logs["eval/best_score"] = best_score
                try:
                    logs["video"] = wandb.Video(
                        frames,
                        fps=30,
                        format="mp4",
                    )
                except Exception as e:
                    print(f"Error creating video: {e}")

                self.prev_eval_timesteps = self.num_timesteps
                # save model
                self.model.save(
                    path.join(
                        self.save_dir,
                        f"model_{self.num_timesteps}_{mean_score:.0f}",
                    )
                )
                if mean_score > self.best_mean_score:
                    self.best_mean_score = mean_score
                    self.model.save(path.join(self.save_dir, "model_best"))

            wandb.log(logs, step=self.num_timesteps)
            self.prev_log_timesteps = self.num_timesteps
        return True
