import copy
import cv2
import pygame
import pymunk
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import namedtuple
from copy import deepcopy
from typing import List

# import imageio
# import matplotlib.font_manager as fm
# import matplotlib.pyplot as plt
# from PIL import Image, ImageDraw, ImageFont
from ding.envs import BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from easydict import EasyDict

# Environment IDs for the four different learning levels
GAME_IDS = {
    1: "suika-game-l1-v0",  # Coordinate-size list with game engine access
    2: "suika-game-l2-v0",  # Coordinate-size list without game engine access
    3: "suika-game-l3-v0",  # Image with game engine access
    4: "suika-game-l4-v0",  # Image without game engine access
}

FruitState = namedtuple("FruitState", ["pos", "radius", "type"])
class Fruit(pymunk.Circle):
    """Represents a fruit in the Suika game"""
    id_cnt = 0

    def __init__(self, pos, type, space):                
        radii = [17, 25, 32, 38, 50, 63, 75, 87, 100, 115, 135]
        friction_fruit = 4
        density = 1
        elasticity = 0.3
        fruit_collection_type = 1
        
        body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        body.position = tuple(pos)
        super().__init__(body=body, radius=radii[type])
        self.density = density
        self.elasticity = elasticity
        self.collision_type = fruit_collection_type
        self.friction = friction_fruit

        self.id = self._new_id()  # Incremental ID for merge precedence
        self.type = type
        self.removed = False

        space.add(self.body, self)  # required to add body first

    @classmethod
    def _new_id(cls):
        cls.id_cnt += 1
        return cls.id_cnt

    @staticmethod
    def draw_state(screen, pos, radius, type, font=None, id=None):
        colors = [
            (245, 0, 0),
            (250, 100, 100),
            (150, 20, 250),
            (250, 210, 10),
            (250, 150, 0),
            (245, 0, 0),
            (250, 250, 100),
            (255, 180, 180),
            (255, 255, 0),
            (100, 235, 10),
            (0, 185, 0),
        ]
        c1 = np.array(colors[type])
        c2 = (c1 * 0.8).astype(int)
        pygame.draw.circle(screen, tuple(c2), pos, radius)
        pygame.draw.circle(screen, tuple(c1), pos, radius * 0.9)

        # Only draw IDs if font is provided - optimization
        if font is not None:
            # Choose a contrasting color (black or white) based on background color brightness
            brightness = sum(c1) / 3
            text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)

            # Render the ID text
            id_text = font.render(f"{id}", 1, text_color)

            # Center the text on the particle
            text_pos = (
                pos[0] - id_text.get_width() // 2,
                pos[1] - id_text.get_height() // 2,
            )
            screen.blit(id_text, text_pos)

    def draw(self, screen, font=None):
        """Draw the fruit on the given Pygame surface"""
        self.draw_state(screen, **self.state._asdict(), font=font, id=self.id)

    @property
    def pos(self):
        return np.array(self.body.position)

    @property
    def state(self):
        return FruitState(self.pos, self.radius, self.type)


N_TYPES = 11
GRAY_STEP = 255 // (N_TYPES + 1)
BG_GRAY = 0
GRAYS = [n * GRAY_STEP for n in range(1, N_TYPES + 1)]
assert len(GRAYS) == N_TYPES

SRC_BOARD_OFFSET = (30, 31)
SRC_WALL_HEIGHT_OFFSET = 130
SRC_BOARD_CROPPED_SIZE = (709, 508)
WALL_THICKNESS = 4

@ENV_REGISTRY.register('suika')
class SuikaEnv(gym.Env):
    # The default_config for suika env.
    config = dict(
        env_id="suika",
        size = (570, 770),
        width = 570,
        height = 770,
        pad = (24, 160),
        wall_thickness = 14,
        a = (24, 160),
        b = (24, 770 - 24),
        c = (570 - 24, 770 - 24),
        d = (570 - 24, 160),
        bg_color = (250, 240, 148),
        wall_color = (250, 190, 58),
        n_types = 11,
        colors = [
            (245, 0, 0),
            (250, 100, 100),
            (150, 20, 250),
            (250, 210, 10),
            (250, 150, 0),
            (245, 0, 0),
            (250, 250, 100),
            (255, 180, 180),
            (255, 255, 0),
            (100, 235, 10),
            (0, 185, 0),
        ],
        radii = [17, 25, 32, 38, 50, 63, 75, 87, 100, 115, 135],
        friction_wall = 10,
        friction_fruit = 4,
        density = 1,  # seems no difference
        elasticity = 0.3,
        impulse = 10000,
        gravity = 2000,  # higher gravity for stronger collisions
        bias = 0.00001,
        points = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66],
        damping = 0.2,  # Lower to decrease the time it takes for space to be stable
        fruit_collection_type = 1,  # 1 for coordinate-size list, 2 for image
        fps = 120,
        physics_step_size = 0.01,
        gameover_min_vec = 0.1,
        n_frames = 4,
        level = 1,
        render_mode = "rgb_array",
        render_fps = 60,
        
        # (str): The mode of the battle. Choices are 'self_play_mode' or 'alpha_beta_pruning'.
        battle_mode='play_with_bot_mode',
        # (str): The mode of Monte Carlo Tree Search. This is only used in AlphaZero.
        battle_mode_in_simulation_env='play_with_bot_mode',
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg
    
    def __init__(self, cfg: dict) -> None:
        default_config = self.default_config()
        default_config.update(cfg)
        self.cfg = default_config
        self._init_flag = False
        self._env_id = self.cfg.env_id
        self.size = self.cfg.size
        self.width = self.cfg.width
        self.height = self.cfg.height
        self.pad = self.cfg.pad
        self.wall_thickness = self.cfg.wall_thickness
        self.a = self.cfg.a
        self.b = self.cfg.b
        self.c = self.cfg.c
        self.d = self.cfg.d
        self.bg_color = self.cfg.bg_color
        self.wall_color = self.cfg.wall_color
        self.n_types = self.cfg.n_types
        self.colors = self.cfg.colors
        self.radii = self.cfg.radii
        self.friction_wall = self.cfg.friction_wall
        self.friction_fruit = self.cfg.friction_fruit
        self.density = self.cfg.density
        self.elasticity = self.cfg.elasticity
        self.impulse = self.cfg.impulse
        self.gravity = self.cfg.gravity
        self.bias = self.cfg.bias
        self.points = self.cfg.points
        self.damping = self.cfg.damping
        self.fruit_collection_type = self.cfg.fruit_collection_type
        self.fps = self.cfg.fps
        self.physics_step_size = self.cfg.physics_step_size
        self.gameover_min_vec = self.cfg.gameover_min_vec
        self.n_frames = self.cfg.n_frames
        self.level = self.cfg.level
        self.render_mode = self.cfg.render_mode
        self.render_fps = self.cfg.render_fps
        
        self.battle_mode = self.cfg.battle_mode
        # The mode of interaction between the agent and the environment.
        assert self.battle_mode in ['self_play_mode', 'play_with_bot_mode', 'eval_mode']
        # The mode of MCTS is only used in AlphaZero.
        self.battle_mode_in_simulation_env = self.cfg.battle_mode_in_simulation_env
        
        self.wrapper_image_size = (96, 96)
        self.resize_ratio = (96 - WALL_THICKNESS) / SRC_BOARD_CROPPED_SIZE[0]
        board_width = (
            round(SRC_BOARD_CROPPED_SIZE[1] * self.resize_ratio) + WALL_THICKNESS * 2
        )  # 74
        horizontal_offset = (self.wrapper_image_size[1] - board_width) // 2
        self.horizontal_wall_thickness = horizontal_offset + WALL_THICKNESS
        # [0, (horizontal_wall_thickness - 1)] for wall
        self.fruit_horizontal_offset = self.horizontal_wall_thickness
        self.wall_height_offset = round(SRC_WALL_HEIGHT_OFFSET * self.resize_ratio)
        
        # set other properties...
        if self.n_frames > self.fps:
            raise ValueError("num_frames > FPS")

        if self.fps % self.n_frames != 0:
            raise ValueError("FPS % num_frames != 0")

        if self.level not in [1, 2, 3, 4]:
            raise ValueError(f"Invalid level: {self.level}. Must be 1, 2, 3, or 4.")

        self.frame_interval = self.fps // self.n_frames
        self.render_interval = 4

        # Image size for image-based representation (levels 3 and 4)
        self.image_size = (self.width, self.height)
        
        # Gym spaces
        # Action space: Continuous value representing x-position (normalized to [0,1])
        self._action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self._reward_range = (0., 1000)
        
        # Set observation space based on level
        self._setup_observation_space()

        # always init
        self._init_render()

        # Initialize other environment variables
        self.rng = np.random.default_rng()
        self.shape_to_particle: dict[pymunk.Shape, Fruit] = {}
        self.fruits: list[Fruit] = []
        self.score = 0
        self.game_over = False
        self.current_step = 0
        self.overflow_counter = 0

        # Setup pymunk space
        self._reset_space()

        # Initialize next particle
        self.next_fruit_type = self._gen_next_fruit_type()
    
    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space
    
    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_range
    
    @property
    def legal_actions(self):
        # get the actual legal actions
        return np.arange(self._action_space.n)
    
    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(evaluator_env_num)]

    def __repr__(self) -> str:
        return "LightZero Suika Env"
    
    def reset(self, start_player_index=0, init_state=None, katago_policy_init=False, katago_game_state=None):
        """Reset the environment to initial state"""
        super().reset()

        # Clear old state
        self.fruits = []
        self.shape_to_particle = {}
        self.score = 0
        self.game_over = False
        self.current_step = 0
        self.overflow_counter = 0
        self.merge_score = 0
        self.merge_count = 0

        # Re-initialize the physics space
        self._reset_space()

        # Generate new first fruits
        self.cur_fruit_type = self._gen_next_fruit_type()
        self.next_fruit_type = self._gen_next_fruit_type()

        # Create initial observation with repeated frames
        boards = [self._get_board() for _ in range(self.n_frames)]
        # if self.level in [1, 2]:
        #     fruit_states = deepcopy(boards)
        # else:
        #     fruit_states = [self._get_list_board() for _ in range(self.n_frames)]
        
        boards = self._transform_board(boards)
        
        observation = self._get_observation(boards)
        # info = self._get_info(fruit_states)
        obs = {
            'observation': observation,
            'action_mask': None,
            'board': observation['boards'],
            'current_player_index': 0,
            'to_play': -1,
        }
        return obs       
        
    
    def step(self, action):
        """
        Take a step in the environment

        Args:
            action: normalized x-position [0,1] where to drop the fruit

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1

        # Process action (map from [0,1] to screen width with padding)
        x_min = self.pad[0] + self.radii[self.next_fruit_type] + self.wall_thickness // 2
        x_max = self.width - x_min
        x_pos = x_min + action[0] * (x_max - x_min)

        # Create and drop new particle
        self.cur_fruit_type = self.next_fruit_type
        self.next_fruit_type = self._gen_next_fruit_type()
        cur_fruit = Fruit(
            (x_pos, self.pad[1] // 2),
            self.cur_fruit_type,
            self.space,
        )
        self.fruits.append(cur_fruit)

        # Run physics for a fixed amount of time
        boards = []
        self.merge_score = 0
        self.merge_count = 0
        for t in range(self.fps):
            self.space.step(self.physics_step_size)

            if t % self.frame_interval == 0:
                boards.append(self._get_board())

            # Render during physics simulation if render_mode is set
            # if self.render_mode == "human" and t % self.render_interval == 0:
            #     self._render_frame_in_pygame_surface(
            #         self.screen,
            #         self.fruits,
            #         self.walls,
            #         human=True,
            #         score=self.score,
            #         level=self.level,
            #         gameover=self.game_over,
            #         **self.fonts,
            #     )
            #     # Display on the screen
            #     pygame.display.flip()
            #     self.clock.tick(self.render_fps)  # Limit program running speed
            #     # Process events to keep the window responsive
            #     for event in pygame.event.get():
            #         if event.type == pygame.QUIT:
            #             pygame.quit()
            #             sys.exit()

            if (
                self.check_game_over()
            ):  # check at beginning so won't collide out of container
                self.game_over = True

        self.score += self.merge_score
        
        boards = self._transform_board(boards)

        observation = self._get_observation(boards)
        # Calculate reward
        reward = self.merge_score  # Reward based on points gained

        # Check termination conditions
        terminated = self.game_over

        if self.level in [1, 2]:
            fruit_states = deepcopy(boards)
        else:
            fruit_states = [self._get_list_board() for _ in range(self.n_frames)]

        info = self._get_info(fruit_states)
        # return observation, reward, terminated, truncated, info
        observation = {
            'observation': observation,
            'action_mask': None,
            'board': observation['boards'],
            'current_player_index': 0,
            'to_play': -1,
        }
        
        return BaseEnvTimestep(observation, reward, terminated, info)

    def render(self, mode: str) -> None:
        """
        Overview:
            Renders the game environment.
        Arguments:
            - mode (:obj:`str`): The rendering mode
        """
        return None
    
    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)
    
    def _setup_observation_space(self):
        """Set up observation space based on level and number of frames"""
        # If levels 1 or 2 (coordinate-size list)
        if self.level in [1, 2]:
            # self._observation_space = spaces.Dict(
            #     {
            #         "boards": spaces.Sequence(
            #             spaces.Sequence(
            #                 spaces.Tuple(
            #                     (
            #                         spaces.Box(0, 1, shape=(2,)),
            #                         spaces.Discrete(max(self.radii) + 1),
            #                         spaces.Discrete(self.n_types),
            #                     )
            #                 )  # TODO: fix
            #             )
            #         ),
            #         "cur_fruit": spaces.Discrete(5),
            #         "next_fruit": spaces.Discrete(5),  # 0-4 for the next fruit types
            #     }
            # )
            self._observation_space = spaces.Dict(
                {
                    "boards": spaces.Box(
                        low=0,
                        high=255,
                        shape=(
                            self.n_frames,
                            1,
                            *self.wrapper_image_size,
                        ),
                        dtype=np.uint8,
                    ),
                    "cur_fruit": spaces.Discrete(5),
                    "next_fruit": spaces.Discrete(5),
                }
            )
        # If levels 3 or 4 (image-based)
        else:
            self._observation_space = spaces.Dict(
                {
                    "boards": spaces.Box(
                        low=0,
                        high=255,
                        shape=(
                            self.n_frames,
                            *self.image_size,
                            3,
                        ),
                        dtype=np.uint8,
                    ),
                    "cur_fruit": spaces.Discrete(5),
                    "next_fruit": spaces.Discrete(5),
                }
            )

    def _reset_space(self):
        """Set up the pymunk physics space and walls"""
        self.space = pymunk.Space()
        self.space.gravity = (0, self.gravity)
        self.space.damping = self.damping
        self.space.collision_bias = self.bias

        def create_wall(a, b):
            """Create a wall segment in the physics space"""
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            shape = pymunk.Segment(body, a, b, self.wall_thickness // 2)
            shape.friction = self.friction_wall
            self.space.add(body, shape)
            return (body, shape)

        # Create walls
        self.walls = []
        self.walls.append(create_wall(self.a, self.b))  # left
        self.walls.append(create_wall(self.b, self.c))  # bottom
        self.walls.append(create_wall(self.c, self.d))  # right

        # Set up collision handler
        handler = self.space.add_collision_handler(
            self.fruit_collection_type, self.fruit_collection_type
        )
        handler.begin = self._collide

    def _init_render(self):
        """Initialize rendering components"""
        pygame.init()
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption(f"Suika Gym - Level {self.level}")
            pygame.font.init()
            self.fonts = dict(
                scorefont=pygame.font.SysFont("monospace", 32),
                overfont=pygame.font.SysFont("monospace", 72),
                idfont=pygame.font.SysFont("monospace", 12),
            )
            self.clock = pygame.time.Clock()
        else:
            self.screen = pygame.Surface((self.width, self.height))
            self.idfont = None

    def _remove_fruit(self, fruit: "Fruit"):
        fruit.removed = True
        self.space.remove(fruit.body, fruit)
        self.fruits.remove(fruit)

    def _collide(self, arbiter, space, data):
        """Collision handler for fruits registered in pymunk"""
        fruit1: Fruit
        fruit2: Fruit
        fruit1, fruit2 = arbiter.shapes
        if fruit1.removed or fruit2.removed:
            # print(f"Collision with removed fruit {fruit1.id} or {fruit2.id}")
            return False  # ignore other collision happens in the same step of merge

        if fruit1.type == fruit2.type:  # merge
            self._remove_fruit(fruit1)
            self._remove_fruit(fruit2)
            # print(f"Merge {fruit1.id}  {fruit2.id}")
            # assume removed fruit information still accessible
            self.merge_score += self.points[fruit1.type]
            self.merge_count += 1
            if fruit1.type != self.n_types - 1:
                new_fruit_pos = fruit1.pos if fruit1.id < fruit2.id else fruit2.pos
                merged_fruit = Fruit(new_fruit_pos, fruit1.type + 1, space)
                self.fruits.append(merged_fruit)
            return False  # ignore collision

            # Apply impulses to nearby particles
            # for p in self.fruits:
            #     vector = p.pos - pn.pos
            #     distance = np.linalg.norm(vector)
            #     if distance < pn.radius + p.radius:
            #         impulse = IMPULSE * vector / (distance**2)
            #         p.body.apply_impulse_at_local_point(tuple(impulse))

        return True  # Return True to allow the collision, False to ignore it

    def _gen_next_fruit_type(self):
        return self.rng.integers(0, 5)

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
    
    def _transform_board(self, boards):
        images = []
        for board in boards:
            image = np.zeros(self.wrapper_image_size, dtype=np.uint8)
            # left wall
            cv2.rectangle(
                image,
                (0, self.wall_height_offset),
                (self.horizontal_wall_thickness - 1, self.wrapper_image_size[0] - 1),
                255,
                -1,
            )
            # right wall
            cv2.rectangle(
                image,
                (
                    self.wrapper_image_size[1] - self.horizontal_wall_thickness,
                    self.wall_height_offset,
                ),
                (self.wrapper_image_size[1] - 1, self.wrapper_image_size[0] - 1),
                255,
                -1,
            )
            # bottom wall
            cv2.rectangle(
                image,
                (0, self.wrapper_image_size[0] - WALL_THICKNESS),
                (self.wrapper_image_size[1], self.wrapper_image_size[0] - 1),
                255,
                -1,
            )
            for pos, r, t in self._transform(board):
                cv2.circle(
                    image, center=pos, radius=int(r), color=GRAYS[t], thickness=-1
                )
            images.append(image)

        return (
            np.array(images)
            .reshape(self.n_frames, *self.wrapper_image_size)  # [C,H,W]
            .astype(np.uint8)
        )
    
    def _get_list_board(self):
        """Get the internal game engine state (positions and radii of particles)"""
        list_state = []
        for p in self.fruits:
            list_state.append(p.state)
        
        return list_state

    def _get_image_board(self):
        return self.render()

    def _get_board(self):
        """Get observation based on level"""
        if self.level in [1, 2]:
            return self._get_list_board()
        else:
            return self._get_image_board()

    def check_game_over(self):
        """Check if game is over (fruit above threshold line with <EPS down vec)"""
        for fruit in self.fruits:
            if (
                fruit.pos[1] - fruit.radius < self.pad[1]
                and fruit.body.velocity.length < self.gameover_min_vec
            ):
                self.overflow_counter += 1
                if self.overflow_counter > 10:
                    return True
                return False
        self.overflow_counter = 0
        return False
    
    # done, winner = env.get_done_winner()
    def get_done_winner(self):
        return self.check_game_over(), 1
    

    def _get_observation(self, boards):
        return {
            "boards": boards,
            "cur_fruit": self.cur_fruit_type,
            "next_fruit": self.next_fruit_type,
        }

    def _get_info(self, fruit_states):
        return {
            "score": self.score,
            "fruit_states": fruit_states,
            "merge_count": self.merge_count,
        }