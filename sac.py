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

# import cv2

config = {
    "env_name": "suika-game-l1-v0",
    "policy_type": "MultiInputPolicy",
    "total_timesteps": 300000,
    "buffer_size": 500000,
    "batch_size": 256,
}


class MyCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        # Extract image sequence shape
        frames_shape = observation_space["boards"].shape
        n_frames = frames_shape[0]

        # CNN to process each frame
        self.cnn = nn.Sequential(
            nn.Conv2d(frames_shape[1], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output dim
        with torch.no_grad():
            sample_frame = torch.zeros((1, *frames_shape[1:]))
            cnn_out_dim = self.cnn(sample_frame).shape[1]

        # Process sequence of frames
        self.frame_proj = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(n_frames * cnn_out_dim, 256),
            nn.ReLU(),
        )

        # Final projection
        self.final = nn.Sequential(nn.Linear(256 + 1 + 1, 256), nn.ReLU())

        self._features_dim = 256

    def forward(self, observations):
        frames = observations["boards"].float() / 255.0  # normalize to [0,1]
        b, t, c, h, w = frames.shape
        frames = frames.view(b * t, c, h, w)
        cnn_out = self.cnn(frames)
        cnn_out = cnn_out.view(b, t * cnn_out.shape[1])
        frame_feat = self.frame_proj(cnn_out)
        # convert one hot back to index
        # if observations["cur_fruit"].shape != torch.Size([1, 5]):
        #    print(observations["cur_fruit"].shape)
        # (64, 1, 5) or (1, 5)
        if len(observations["cur_fruit"].shape) == 2:
            observations["cur_fruit"] = observations["cur_fruit"].unsqueeze(0)
            observations["next_fruit"] = observations["next_fruit"].unsqueeze(0)
        cur_fruit = (
            torch.argmax(observations["cur_fruit"], dim=-1)
        ) + 1  # +1 to align with gray values
        next_fruit = torch.argmax(observations["next_fruit"], dim=-1) + 1

        return self.final(
            torch.cat(
                [
                    frame_feat,
                    cur_fruit,
                    next_fruit,
                ],
                dim=1,
            )
        )


class WandbLoggingCallback(BaseCallback):
    def __init__(self, eval_env, log_interval=500, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.interval = log_interval
        self.eval_interval = log_interval * 10
        self.ep_reward = 0
        self.rewards = deque(maxlen=100)

    def _on_step(self) -> bool:
        def evaluate():
            env = self.eval_env
            obs, info = env.reset()
            done = False
            score = 0
            frames = []
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                score += reward
                frames += env.unwrapped.render_states(states=info["fruit_states"])
            return score, frames

        self.ep_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.rewards.append(self.ep_reward)
            self.ep_reward = 0

        if (self.num_timesteps + 1) % self.interval == 0:
            logs = {
                "train_score": np.mean(self.rewards) if len(self.rewards) > 0 else 0,
                "actor_loss": self.logger.name_to_value["train/actor_loss"],
                "critic_loss": self.logger.name_to_value["train/critic_loss"],
                "ent_coef": self.logger.name_to_value["train/ent_coef"],
            }
            if (self.num_timesteps + 1) % self.eval_interval == 0:
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
            wandb.log(logs, step=self.num_timesteps)
        return True


run = wandb.init(
    project="suika-sb3-sac",
    name=datetime.datetime.now().strftime("%m-%d_%H-%M"),
    config=config,
    settings=wandb.Settings(x_disable_stats=True),
)


def make_env():
    env = gym.make(config["env_name"], render_mode="rgb_array")
    env = CoordSizeToImage(env=env)
    return env


env = make_env()

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
)
model.learn(
    total_timesteps=config["total_timesteps"],
    log_interval=10,  # episode
    progress_bar=True,
    callback=WandbLoggingCallback(make_env()),
)
# model.save("sac_model")
run.finish()
