from suika_gym import SuikaEnv
from wrappers import CoordSizeToImage
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import SAC
from wandb.integration.sb3 import WandbCallback
import wandb
import torch
import torch.nn as nn
import numpy as np
import datetime

config = {
    "policy_type": "MultiInputPolicy",
    "total_timesteps": 300000,
    "buffer_size": 500000,
    "batch_size": 512,
}

run = wandb.init(
    project="suika-sb3-sac",
    name=datetime.datetime.now().strftime("%m-%d_%H-%M"),
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    settings=wandb.Settings(x_disable_stats=True),
)


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


env = CoordSizeToImage(env=SuikaEnv())

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
    log_interval=10,
    progress_bar=True,
    callback=WandbCallback(
        model_save_freq=1000,
        model_save_path=f"weights/{run.id}",
        gradient_save_freq=1000,
        verbose=2,
    ),
)
# model.save("sac_model")
run.finish()
