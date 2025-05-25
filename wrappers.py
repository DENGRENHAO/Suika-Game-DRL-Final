import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2

N_TYPES = 11
GRAY_STEP = 255 // (N_TYPES + 1)
BG_GRAY = 0
GRAYS = [n * GRAY_STEP for n in range(1, N_TYPES)]

SRC_BOARD_OFFSET = (30, 31)
SRC_WALL_HEIGHT_OFFSET = 130
SRC_BOARD_CROPPED_SIZE = (709, 508)
WALL_THICKNESS = 4

n_frames = 8


class CoordSizeToImage(gym.ObservationWrapper):
    def __init__(self, env, image_size=(96, 96)):
        gym.ObservationWrapper.__init__(self, env)
        self.image_size = image_size
        self.resize_ratio = (96 - WALL_THICKNESS) / SRC_BOARD_CROPPED_SIZE[0]
        board_width = (
            round(SRC_BOARD_CROPPED_SIZE[1] * self.resize_ratio) + WALL_THICKNESS * 2
        )  # 74
        horizontal_offset = (image_size[1] - board_width) // 2
        self.horizontal_wall_thickness = horizontal_offset + WALL_THICKNESS
        # [0, (horizontal_wall_thickness - 1)] for wall
        self.fruit_horizontal_offset = self.horizontal_wall_thickness
        self.wall_height_offset = round(SRC_WALL_HEIGHT_OFFSET * self.resize_ratio)
        self.observation_space = spaces.Dict(
            {
                "boards": spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        n_frames,
                        1,
                        *image_size,
                    ),
                    dtype=np.uint8,
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
                    + self.fruit_horizontal_offset,
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
            # left wall
            cv2.rectangle(
                image,
                (0, self.wall_height_offset),
                (self.horizontal_wall_thickness - 1, self.image_size[0] - 1),
                255,
                -1,
            )
            # right wall
            cv2.rectangle(
                image,
                (
                    self.image_size[1] - self.horizontal_wall_thickness,
                    self.wall_height_offset,
                ),
                (self.image_size[1] - 1, self.image_size[0] - 1),
                255,
                -1,
            )
            # bottom wall
            cv2.rectangle(
                image,
                (0, self.image_size[0] - WALL_THICKNESS),
                (self.image_size[1], self.image_size[0] - 1),
                255,
                -1,
            )
            for pos, r, t in self._transform(board):
                cv2.circle(
                    image, center=pos, radius=int(r), color=GRAYS[t], thickness=-1
                )
            images.append(image)

        observation["boards"] = (
            np.array(images)
            .reshape(n_frames, 1, *self.image_size)  # [B,C,H,W]
            .astype(np.uint8)
        )
        # observation["boards"] = (
        #    torch.from_numpy(np.array(images))
        #    .unsqueeze(-1)
        #    .permute(0, 3, 1, 2)  # [B,C,H,W]
        # )
        #
        ## not needed in SB3 because it converts to one-hot by value
        # observation["cur_fruit"] = torch.tensor(
        #    observation["cur_fruit"], dtype=torch.int8
        # )
        # observation["next_fruit"] = torch.tensor(
        #    observation["next_fruit"], dtype=torch.int8
        # )

        return observation
