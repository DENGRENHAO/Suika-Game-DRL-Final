import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import torch

N_TYPES = 11
GRAY_STEP = 255 // (N_TYPES + 1)
BG_GRAY = 0
GRAYS = [n * GRAY_STEP for n in range(1, N_TYPES)]

SRC_BOARD_OFFSET = (30, 31)
SRC_WALL_HEIGHT_OFFSET = 130
SRC_BOARD_CROPPED_SIZE = (709, 508)
WALL_THICKNESS = 4

n_frames = 8


def normalize_images(images):
    return [image.astype(np.float32) / 255.0 for image in images]


class CoordSizeToImage(gym.ObservationWrapper):
    def __init__(self, env, image_size=(96, 96)):
        gym.ObservationWrapper.__init__(self, env)
        self.image_size = image_size
        self.resize_ratio = (96 - WALL_THICKNESS) / SRC_BOARD_CROPPED_SIZE[0]
        self.wall_left_offset = (
            round((image_size[1] - SRC_BOARD_CROPPED_SIZE[1] * self.resize_ratio) / 2)
            - WALL_THICKNESS
        )
        self.wall_right_offset = (
            image_size[1] - 1 - self.wall_left_offset - WALL_THICKNESS
        )
        self.wall_height_offset = round(SRC_WALL_HEIGHT_OFFSET * self.resize_ratio)
        self.fruit_left_offset = self.wall_left_offset + WALL_THICKNESS
        self.observation_space = spaces.Dict(
            {
                "boards": spaces.Box(
                    low=0,
                    high=1,
                    shape=(
                        n_frames,
                        1,
                        *image_size,
                    ),
                    dtype=np.float32,
                ),
                "cur_fruit": spaces.Discrete(5),
                "next_fruit": spaces.Discrete(5),
            }
        )

    def _transform(self, board):
        return [
            (
                (
                    round((pos[0] - SRC_BOARD_OFFSET[0]) * self.resize_ratio)
                    + self.fruit_left_offset,
                    round((pos[1] - SRC_BOARD_OFFSET[1]) * self.resize_ratio),
                ),
                r * self.resize_ratio,
                t,
            )
            for pos, r, t in board
        ]

    def observation(self, observation):
        images = []
        for board in observation["boards"]:
            image = np.zeros(self.image_size, dtype=np.uint8)
            cv2.rectangle(
                image,
                (0, self.wall_height_offset),
                (self.wall_left_offset + WALL_THICKNESS, self.image_size[0]),
                255,
                -1,
            )
            cv2.rectangle(
                image,
                (self.wall_right_offset, self.wall_height_offset),
                (self.image_size[1], self.image_size[0]),
                255,
                -1,
            )
            cv2.rectangle(
                image,
                (self.wall_left_offset, self.image_size[1] - WALL_THICKNESS),
                (self.wall_right_offset + WALL_THICKNESS, self.image_size[1]),
                255,
                -1,
            )
            for pos, r, t in self._transform(board):
                cv2.circle(
                    image, center=pos, radius=int(r), color=GRAYS[t], thickness=-1
                )
            images.append(image)
        images = normalize_images(images)
        # To tensor for RL library
        observation["boards"] = (
            torch.from_numpy(np.array(images))
            .unsqueeze(-1)
            .permute(0, 3, 1, 2)  # [B,C,H,W]
        )
        
        # not needed in SB3 because it converts to one-hot by value
        observation["cur_fruit"] = torch.tensor(
            observation["cur_fruit"], dtype=torch.int8
        )
        observation["next_fruit"] = torch.tensor(
            observation["next_fruit"], dtype=torch.int8
        )

        return observation


class NormalizeFrame(gym.ObservationWrapper):
    def __init__(self, env, image_size):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Dict(
            {
                "boards": spaces.Box(
                    low=0,
                    high=1,
                    shape=(
                        n_frames,
                        *image_size,
                        3,
                    ),
                    dtype=np.float32,
                ),
                "cur_fruit": spaces.Discrete(5),
                "next_fruit": spaces.Discrete(5),
            }
        )

    def observation(self, observation):
        observation["frames"] = normalize_images(observation["frames"])
        return observation
