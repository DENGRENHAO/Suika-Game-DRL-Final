import sys
import numpy as np
import pygame
import pymunk
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from collections import namedtuple

SIZE = WIDTH, HEIGHT = np.array([570, 770])
PAD = (24, 160)
WALL_THICKNESS = 14
A = (PAD[0], PAD[1])
B = (PAD[0], HEIGHT - PAD[0])
C = (WIDTH - PAD[0], HEIGHT - PAD[0])
D = (WIDTH - PAD[0], PAD[1])
BG_COLOR = (250, 240, 148)
W_COLOR = (250, 190, 58)
N_TYPES = 11
COLORS = [
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
RADII = [17, 25, 32, 38, 50, 63, 75, 87, 100, 115, 135]
assert len(COLORS) == len(RADII) == N_TYPES, "Number of colors or radii is not N_TYPES"
FRICTION_WALL = 10
FRICTION_FRUIT = 4
DENSITY = 1  # seems no difference
ELASTICITY = 0.3
IMPULSE = 10000
GRAVITY = 2000  # higher gravity for stronger collisions
BIAS = 0.00001
POINTS = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66]
DAMPING = 0.2  # Lower to decrease the time it takes for space to be stable
FRUIT_COLLECTION_TYPE = 1

FPS = 120
PHYSICS_STEP_SIZE = 0.01
GAMEOVER_MIN_VEC = 0.1

# Environment IDs for the four different learning levels
GAME_IDS = {
    1: "suika-game-l1-v0",  # Coordinate-size list with game engine access
    2: "suika-game-l2-v0",  # Coordinate-size list without game engine access
    3: "suika-game-l3-v0",  # Image with game engine access
    4: "suika-game-l4-v0",  # Image without game engine access
}


class SuikaEnv(gym.Env):
    def __init__(self, n_frames=8, level=1, render_mode=None, render_fps=60):
        """
        Initialize the Suika game environment

        Args:
            render_mode: "human" for window display
            num_frames: Number of intermediate frames to capture, <=FPS && |FPS. (Ideally always provide FPS, but generate on demand for speed.)
            fps: Frames per step for physics simulation
            level: Learning level (1-4)
                   1: Coordinate-size list with game engine access
                   2: Coordinate-size list without game engine access
                   3: Image with game engine access
                   4: Image without game engine access
        """
        if n_frames > FPS:
            raise ValueError("num_frames > FPS")

        if FPS % n_frames != 0:
            raise ValueError("FPS % num_frames != 0")

        if level not in [1, 2, 3, 4]:
            raise ValueError(f"Invalid level: {self.level}. Must be 1, 2, 3, or 4.")

        self.render_fps = render_fps
        self.level = level
        self.n_frames = n_frames  # Number of intermediate frames to collect
        self.frame_interval = FPS // n_frames
        self.render_interval = 4

        # Grid size for coordinate representation
        self.grid_size = (WIDTH // 10, HEIGHT // 10)

        # Image size for image-based representation (levels 3 and 4)
        self.image_size = (WIDTH, HEIGHT)

        # Gym spaces
        # Action space: Continuous value representing x-position (normalized to [0,1])
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Set observation space based on level
        self._setup_observation_space()

        # Initialize pygame if needed
        self.render_mode = render_mode
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
        self.next_fruit_type = self.rng.integers(0, 5)
        self.cur_fruit_x = WIDTH // 2

    def _setup_observation_space(self):
        """Set up observation space based on level and number of frames"""
        # If levels 1 or 2 (coordinate-size list)
        if self.level in [1, 2]:
            self.observation_space = spaces.Dict(
                {
                    "boards": spaces.Sequence(
                        spaces.Sequence(
                            spaces.Tuple(
                                (
                                    spaces.Box(0, 1, shape=(2,)),
                                    spaces.Discrete(max(RADII) + 1),
                                    spaces.Discrete(N_TYPES),
                                )
                            )  # TODO: fix
                        )
                    ),
                    "next_fruit": spaces.Discrete(5),  # 0-4 for the next fruit types
                }
            )
        # If levels 3 or 4 (image-based)
        else:
            self.observation_space = spaces.Dict(
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
                    "next_fruit": spaces.Discrete(5),
                }
            )

    def _reset_space(self):
        """Set up the pymunk physics space and walls"""
        self.space = pymunk.Space()
        self.space.gravity = (0, GRAVITY)
        self.space.damping = DAMPING
        self.space.collision_bias = BIAS

        def create_wall(a, b):
            """Create a wall segment in the physics space"""
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            shape = pymunk.Segment(body, a, b, WALL_THICKNESS // 2)
            shape.friction = FRICTION_WALL
            self.space.add(body, shape)
            return (body, shape)

        # Create walls
        self.walls = []
        self.walls.append(create_wall(A, B))  # left
        self.walls.append(create_wall(B, C))  # bottom
        self.walls.append(create_wall(C, D))  # right

        # Set up collision handler
        handler = self.space.add_collision_handler(
            FRUIT_COLLECTION_TYPE, FRUIT_COLLECTION_TYPE
        )
        handler.begin = self._collide

    def _init_render(self):
        """Initialize rendering components"""
        pygame.init()
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption(f"Suika Gym - Level {self.level}")
            pygame.font.init()
            self.scorefont = pygame.font.SysFont("monospace", 32)
            self.overfont = pygame.font.SysFont("monospace", 72)
            self.idfont = pygame.font.SysFont("monospace", 12)
            self.clock = pygame.time.Clock()
        else:
            self.screen = pygame.Surface((WIDTH, HEIGHT))

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
            self.collision_score += POINTS[fruit1.type]
            if fruit1.type != N_TYPES - 1:
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

    def _get_board(self):
        def get_image_board():
            return self.render()

        def get_list_board():
            """Get the internal game engine state (positions and radii of particles)"""
            list_state = []
            for p in self.fruits:
                list_state.append(p.state)
            return list_state

        """Get observation based on level"""
        if self.level in [1, 2]:
            return get_list_board()
        else:
            return get_image_board()

    def check_game_over(self):
        """Check if game is over (fruit above threshold line with <EPS down vec)"""
        for fruit in self.fruits:
            if (
                fruit.pos[1] - fruit.radius < PAD[1]
                and fruit.body.velocity.length < GAMEOVER_MIN_VEC
            ):
                self.overflow_counter += 1
                if self.overflow_counter > 10:
                    return True
                return False
        self.overflow_counter = 0
        return False

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)

        # Clear old state
        self.fruits = []
        self.shape_to_particle = {}
        self.score = 0
        self.game_over = False
        self.current_step = 0
        self.overflow_counter = 0

        # Re-initialize the physics space
        self._reset_space()

        # Generate new first fruit
        self.next_fruit_type = self.rng.integers(0, 5)
        self.cur_fruit_x = WIDTH // 2

        # Create initial observation with repeated frames
        boards = [self._get_board() for _ in range(self.n_frames)]
        observation = {
            "boards": boards,
            "next_fruit": self.next_fruit_type,
        }

        info = {"score": self.score}
        return observation, info

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
        x_min = PAD[0] + RADII[self.next_fruit_type] + WALL_THICKNESS // 2
        x_max = WIDTH - x_min
        x_pos = x_min + action[0] * (x_max - x_min)
        self.cur_fruit_x = x_pos

        # Create and drop new particle
        old_score = self.score
        new_fruit = Fruit(
            (x_pos, PAD[1] // 2),
            self.next_fruit_type,
            self.space,
        )
        self.fruits.append(new_fruit)

        # Run physics for a fixed amount of time
        boards = []
        self.collision_score = 0
        for t in range(FPS):
            self.space.step(PHYSICS_STEP_SIZE)

            if t % self.frame_interval == 0:
                boards.append(self._get_board())

            # Render during physics simulation if render_mode is set
            if self.render_mode == "human" and t % self.render_interval == 0:
                self._render_frame_in_pygame_surface(human=True)
                # Display on the screen
                pygame.display.flip()
                self.clock.tick(self.render_fps)  # Limit program running speed
                # Process events to keep the window responsive
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

            if (
                self.check_game_over()
            ):  # check at beginning so won't collide out of container
                self.game_over = True

        self.score += self.collision_score

        # Generate next fruit
        self.next_fruit_type = self.rng.integers(0, 5)

        # Calculate reward
        reward = self.score - old_score  # Reward based on points gained

        # Check termination conditions
        terminated = self.game_over
        truncated = False

        # Get observation
        observation = {"boards": boards, "next_fruit": self.next_fruit_type}

        # Prepare info dict based on level
        info = {"score": self.score}

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment"""
        self._render_frame_in_pygame_surface()
        scaled_surface = pygame.transform.scale(self.screen, self.image_size)

        # Convert to numpy array [C, H, W] pytorch format
        img_array = np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_surface)), axes=(2, 1, 0)
        )

        return img_array

    def _render_frame_in_pygame_surface(self, human=False):
        """Render a single frame of the environment"""
        # redraw everything if human is fine, won't affect training

        def draw_next_particle():
            """Draw the next particle indicator"""
            n = self.next_fruit_type
            radius = RADII[n]
            c1 = np.array(COLORS[n])
            c2 = (c1 * 0.8).astype(int)
            pygame.draw.circle(
                self.screen, tuple(c2), (self.cur_fruit_x, PAD[1] // 2), radius
            )
            pygame.draw.circle(
                self.screen,
                tuple(c1),
                (self.cur_fruit_x, PAD[1] // 2),
                radius * 0.9,
            )

        # Fill background
        self.screen.fill(BG_COLOR)

        # Draw walls
        for wall in self.walls:
            pygame.draw.line(self.screen, W_COLOR, wall[1].a, wall[1].b, WALL_THICKNESS)

        # Draw particles - optimization: only draw active ones
        for p in self.fruits:
            p.draw(self.screen, label=self.idfont if human else None)

        if not self.game_over:
            draw_next_particle()
        elif human:  # Draw game over
            game_over_label = self.overfont.render("Game Over!", 1, (0, 0, 0))
            self.screen.blit(game_over_label, PAD)

        if human:
            # Draw score
            score_label = self.scorefont.render(f"Score: {self.score}", 1, (0, 0, 0))
            self.screen.blit(score_label, (10, 10))

            # Draw level indicator
            level_label = self.scorefont.render(f"Level: {self.level}", 1, (0, 0, 0))
            self.screen.blit(level_label, (10, 50))

    def close(self):
        """Clean up resources"""
        pygame.quit()


FruitState = namedtuple("FruitState", ["pos", "radius", "type"])


class Fruit(pymunk.Circle):
    """Represents a fruit in the Suika game"""

    id_cnt = 0

    def __init__(self, pos, type, space):
        if type >= N_TYPES:
            raise ValueError(f"Invalid fruit type {type}")
        body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        body.position = tuple(pos)
        super().__init__(body=body, radius=RADII[type])
        self.density = DENSITY
        self.elasticity = ELASTICITY
        self.collision_type = FRUIT_COLLECTION_TYPE
        self.friction = FRICTION_FRUIT

        self.id = self._new_id()  # Incremental ID for merge precedence
        self.type = type
        self.removed = False

        space.add(self.body, self)  # required to add body first

    @classmethod
    def _new_id(cls):
        cls.id_cnt += 1
        return cls.id_cnt

    def draw(self, screen, label=None):
        c1 = np.array(COLORS[self.type])
        c2 = (c1 * 0.8).astype(int)
        position = self.body.position
        pygame.draw.circle(screen, tuple(c2), position, self.radius)
        pygame.draw.circle(screen, tuple(c1), position, self.radius * 0.9)

        # Only draw IDs if font is provided - optimization
        if label is not None:
            # Choose a contrasting color (black or white) based on background color brightness
            brightness = sum(c1) / 3
            text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)

            # Render the ID text
            id_text = label.render(f"{self.id}", 1, text_color)

            # Center the text on the particle
            text_pos = (
                position[0] - id_text.get_width() // 2,
                position[1] - id_text.get_height() // 2,
            )
            screen.blit(id_text, text_pos)

    @property
    def pos(self):
        return np.array(self.body.position)

    @property
    def state(self):
        return FruitState(self.pos, self.radius, self.type)


# Register the environments with Gymnasium
def register_envs():
    for level, game_id in GAME_IDS.items():
        try:
            register(
                id=game_id, entry_point="suika_gym:SuikaEnv", kwargs={"level": level}
            )
        except Exception as e:
            print(f"Registration note for level {level}: {e}")


register_envs()
