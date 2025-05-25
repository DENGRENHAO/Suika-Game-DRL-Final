import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn


class MyCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim=features_dim)

        # Extract image sequence shape
        frames_shape = observation_space["boards"].shape
        n_frames = frames_shape[0]

        # CNN to process each frame
        self.cnn = nn.Sequential(
            nn.Conv2d(frames_shape[1], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),  # start_dim=1 by default
        )

        for layer in self.cnn:
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.xavier_uniform_(layer.weight)

        # Compute CNN output dim
        with torch.no_grad():
            sample_frame = torch.zeros((1, *frames_shape[1:]))
            cnn_out_dim = self.cnn(sample_frame).shape[1]

        # Process sequence of frames
        self.frame_proj = nn.Sequential(
            nn.Linear(n_frames * cnn_out_dim, features_dim),
            nn.ReLU(),
        )

        # Final projection
        self.final = nn.Sequential(
            nn.Linear(features_dim + 1 + 1, features_dim), nn.ReLU()
        )

        self._features_dim = features_dim

    def forward(self, observations):
        frames = observations["boards"].float() / 255.0  # normalize to [0,1]
        b, t, c, h, w = frames.shape
        frames = frames.view(b * t, c, h, w)
        cnn_out = self.cnn(frames)  # (b * t, cnn_out_dim)
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
            torch.argmax(observations["cur_fruit"], dim=-1) + 1
        ) / 5  # +1 to align with gray values, and normalize to [0,1]
        next_fruit = (torch.argmax(observations["next_fruit"], dim=-1) + 1) / 5

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
