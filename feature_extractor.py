import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn


class MyCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 64 + 2):
        super().__init__(observation_space, features_dim=features_dim)

        # Extract image sequence shape
        frames_shape = observation_space["boards"].shape
        n_channels = frames_shape[0]  # n_frames

        # CNN to process each frame
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),  # start_dim=1 by default
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output dim
        with torch.no_grad():
            sample_input = torch.zeros((1, *frames_shape))
            cnn_out_dim = self.cnn(sample_input).shape[1]  # 4800

        # Process sequence of frames
        self.frame_proj = nn.Sequential(
            nn.Linear(cnn_out_dim, 64),
            nn.ReLU(),
        )

        self._features_dim = features_dim

        for layer in self.cnn:
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, observations):
        frames = observations["boards"].float() / 255.0  # normalize to [0,1]
        cnn_out = self.cnn(frames)  # [B, 4800]
        projected = self.frame_proj(cnn_out)  # [B, 64]
        # convert one hot back to index
        # if observations["cur_fruit"].shape != torch.Size([1, 5]):
        #    print(observations["cur_fruit"].shape)
        # (64, 1, 5) or (1, 5) (batched or single)
        cur_fruit = observations["cur_fruit"]
        cur_fruit = cur_fruit.view(cur_fruit.shape[0], cur_fruit.shape[-1])
        next_fruit = observations["next_fruit"]
        next_fruit = next_fruit.view(next_fruit.shape[0], next_fruit.shape[-1])
        # +1 to align with gray values, and normalize to [0,1]
        cur_fruit = (torch.argmax(cur_fruit, dim=-1, keepdim=True) + 1) / 5
        next_fruit = (torch.argmax(next_fruit, dim=-1, keepdim=True) + 1) / 5
        # print(
        #    f"cur_fruit: {observations['cur_fruit'].shape}, next_fruit: {observations['next_fruit'].shape}"
        # )

        return torch.cat(
            [
                projected,
                cur_fruit,
                next_fruit,
            ],
            dim=1,
        )
