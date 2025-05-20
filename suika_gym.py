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
DENSITY = 0.001
ELASTICITY = 0.1
IMPULSE = 10000
GRAVITY = 5000  # higher gravity for stronger collisions
BIAS = 0.00001
POINTS = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66]
DAMPING = 0.2  # Lower to decrease the time it takes for space to be stable
FRUIT_COLLECTION_TYPE = 1

FPS = 120
PHYSICS_STEP_SIZE = 0.01

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
        self._pygame_initialized = False
        if self.render_mode is not None:
            self._init_render()

        # Initialize other environment variables
        self.rng = np.random.default_rng()
        self.shape_to_particle: dict[pymunk.Shape, Fruit] = {}
        self.fruits: list[Fruit] = []
        self.score = 0
        self.game_over = False
        self.current_step = 0

        # Setup pymunk space
        self._reset_space()

        # Initialize next particle
        self.next_particle_type = self.rng.integers(0, 5)
        self.next_particle_x = WIDTH // 2

    def _setup_observation_space(self):
        """Set up observation space based on level and number of frames"""
        # If levels 1 or 2 (coordinate-size list)
        if self.level in [1, 2]:
            self.observation_space = spaces.Dict(
                {
                    "board": spaces.Sequence(
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
                    "image": spaces.Box(
                        low=0,
                        high=255,
                        shape=(
                            self.image_size[1],
                            self.image_size[0],
                            3,
                            self.n_frames,
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
            shape.friction = 10
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
        # Store collision data directly in the environment since we can't set handler.data
        self.collision_score = 0

    def _init_render(self):
        """Initialize rendering components"""
        if not self._pygame_initialized:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
                pygame.display.set_caption(f"Suika Gym - Level {self.level}")
            else:
                self.screen = pygame.Surface((WIDTH, HEIGHT))

            self.clock = pygame.time.Clock()
            pygame.font.init()
            self.scorefont = pygame.font.SysFont("monospace", 32)
            self.overfont = pygame.font.SysFont("monospace", 72)
            self.idfont = pygame.font.SysFont("monospace", 12)
            self._pygame_initialized = True

    def _collide(self, arbiter, space, data):
        """Collision handler for fruits"""
        fruit1: Fruit
        fruit2: Fruit
        fruit1, fruit2 = arbiter.shapes

        same_type = bool(fruit1.type == fruit2.type)
        cond = not same_type
        fruit1.has_collided = cond
        fruit2.has_collided = cond

        if same_type:  # Same fruit type, merge them
            new_particle = self._merge_fruits(fruit1, fruit2, space)
            if new_particle is not None:
                self.fruits.append(new_particle)
            self.collision_score += POINTS[fruit1.type]

        return cond  # Return True to allow the collision, False to ignore it

    def _merge_fruits(self, fruit1: "Fruit", fruit2: "Fruit", space):
        """Resolve collision between two identical fruits"""
        fruit1.kill(space)
        fruit2.kill(space)

        # Create new merged particle
        if fruit1.type == N_TYPES - 1:
            return None

        new_pos = fruit1.pos if fruit1.id < fruit2.id else fruit2.pos
        pn = Fruit(new_pos, fruit1.type + 1, space)

        # Apply impulses to nearby particles
        # for p in self.fruits:
        #     if p.alive:
        #         vector = p.pos - pn.pos
        #         distance = np.linalg.norm(vector)
        #         if distance < pn.radius + p.radius:
        #             impulse = IMPULSE * vector / (distance**2)
        #             p.body.apply_impulse_at_local_point(tuple(impulse))

        return pn

    def _get_image_board(self):
        """Get image-based observation (for levels 3 and 4)"""
        # Ensure render component is initialized
        if not hasattr(self, "screen"):
            if not self._pygame_initialized:
                # Initialize a hidden screen for rendering
                pygame.init()
                self.screen = pygame.Surface((WIDTH, HEIGHT))
                pygame.font.init()
                self.idfont = None  # Don't show IDs in the observation
                self._pygame_initialized = True

        # Render the game state to the surface
        self.screen.fill(BG_COLOR)

        # Draw next particle indicator
        if not self.game_over:
            self._draw_next_particle()

        # Draw walls
        for wall in self.walls:
            pygame.draw.line(self.screen, W_COLOR, wall[1].a, wall[1].b, WALL_THICKNESS)

        # Draw particles
        active_particles = [p for p in self.fruits if p.alive]
        for p in active_particles:
            p.draw(self.screen, None)  # Don't draw IDs in the observation

        # Scale the surface to the desired size
        scaled_surface = pygame.transform.scale(self.screen, self.image_size)

        # Convert to numpy array
        img_array = np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_surface)), axes=(1, 0, 2)
        )

        return img_array

    def _get_list_board(self):
        """Get the internal game engine state (positions and radii of particles)"""
        list_state = []
        for p in self.fruits:
            list_state.append(p.state)
        return list_state

    def _get_board(self):
        """Get observation based on level"""
        if self.level in [1, 2]:
            return self._get_list_board()
        else:
            return self._get_image_board()

    def check_game_over(self):
        """Check if game is over (fruit above threshold line)"""
        for p in self.fruits:
            if p.alive and p.has_collided and p.pos[1] < PAD[1]:
                return True
        return False

    def _clean_invalid_particles(self):
        """Clean up particles with invalid positions"""
        to_remove = []
        for p in self.fruits:
            if p.alive and np.isnan(p.pos).any():
                p.kill(self.space)
                to_remove.append(p)

        # Remove killed particles from the list
        for p in to_remove:
            try:
                self.fruits.remove(p)
            except ValueError:
                pass

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)

        # Clear old state
        self.fruits = []
        self.shape_to_particle = {}
        self.score = 0
        self.game_over = False
        self.current_step = 0

        # Re-initialize the physics space
        self._reset_space()

        # Generate new first fruit
        self.next_particle_type = self.rng.integers(0, 5)
        self.next_particle_x = WIDTH // 2

        # Create initial observation with repeated frames
        if self.level in [1, 2]:
            observation = {
                "board": [[] for _ in range(self.n_frames)],
                "next_fruit": self.next_particle_type,
            }
        else:
            # Create an empty image
            empty_image = np.zeros(
                (self.image_size[1], self.image_size[0], 3), dtype=np.uint8
            )
            if self.render_mode is not None:
                # Use the actual rendered image instead of empty
                empty_image = self._get_image_board()
            stacked_images = np.stack([empty_image] * self.n_frames, axis=3)
            observation = {
                "image": stacked_images,
                "next_fruit": self.next_particle_type,
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
        x_min = PAD[0] + RADII[self.next_particle_type] + WALL_THICKNESS // 2
        x_max = WIDTH - x_min
        x_pos = x_min + action[0] * (x_max - x_min)

        # Create and drop new particle
        old_score = self.score
        new_particle = Fruit(
            (x_pos, PAD[1] // 2),
            self.next_particle_type,
            self.space,
        )
        self.fruits.append(new_particle)

        # Run physics for a fixed amount of time
        boards = []
        self.collision_score = 0
        for t in range(FPS):
            self.space.step(PHYSICS_STEP_SIZE)
            self._clean_invalid_particles()

            if t % self.frame_interval == 0:
                boards.append(self._get_board())

            # Render during physics simulation if render_mode is set
            if self.render_mode is not None and t % self.render_interval == 0:
                self._render_frame()

                if self.render_mode == "human":
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
        self.next_particle_type = self.rng.integers(0, 5)

        # Calculate reward
        reward = self.score - old_score  # Reward based on points gained

        # Check termination conditions
        terminated = self.game_over
        truncated = False

        # Get observation
        if self.level in [1, 2]:
            observation = {"board": boards, "next_fruit": self.next_particle_type}
        else:
            observation = {
                "image": boards,
                "next_fruit": self.next_particle_type,
            }

        # Prepare info dict based on level
        info = {"score": self.score}

        return observation, reward, terminated, truncated, info

    def _draw_next_particle(self):
        """Draw the next particle indicator"""
        n = self.next_particle_type
        radius = RADII[n]
        c1 = np.array(COLORS[n])
        c2 = (c1 * 0.8).astype(int)
        pygame.draw.circle(
            self.screen, tuple(c2), (self.next_particle_x, PAD[1] // 2), radius
        )
        pygame.draw.circle(
            self.screen,
            tuple(c1),
            (self.next_particle_x, PAD[1] // 2),
            radius * 0.9,
        )

    def render(self):
        """Render the environment"""
        if self.render_mode is None:
            return

        return self._render_frame()

    def _render_frame(self):
        """Render a single frame of the environment"""
        if not hasattr(self, "screen"):
            return None

        # Fill background
        self.screen.fill(BG_COLOR)

        # Draw next particle indicator
        if not self.game_over:
            self._draw_next_particle()

        # Draw walls
        for wall in self.walls:
            pygame.draw.line(self.screen, W_COLOR, wall[1].a, wall[1].b, WALL_THICKNESS)

        # Draw particles - optimization: only draw active ones
        active_particles = [p for p in self.fruits if p.alive]
        for p in active_particles:
            p.draw(self.screen, self.idfont)

        # Draw score
        score_label = self.scorefont.render(f"Score: {self.score}", 1, (0, 0, 0))
        self.screen.blit(score_label, (10, 10))

        # Draw level indicator
        level_label = self.scorefont.render(f"Level: {self.level}", 1, (0, 0, 0))
        self.screen.blit(level_label, (10, 50))

        # Draw game over
        if self.game_over:
            game_over_label = self.overfont.render("Game Over!", 1, (0, 0, 0))
            self.screen.blit(game_over_label, PAD)

        # Display or return the screen
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.render_fps)

        return None

    def close(self):
        """Clean up resources"""
        if self._pygame_initialized and self.render_mode == "human":
            self._pygame_initialized = False
            pygame.quit()
            sys.exit()


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
        self.friction = 2

        self.id = self._new_id()  # Incremental ID for merge precedence
        self.type = type
        self.has_collided = False

        space.add(self.body, self)  # required to add body first
        self.alive = True

    @classmethod
    def _new_id(cls):
        cls.id_cnt += 1
        return cls.id_cnt

    def draw(self, screen, font):
        if self.alive:
            c1 = np.array(COLORS[self.type])
            c2 = (c1 * 0.8).astype(int)
            position = self.body.position
            pygame.draw.circle(screen, tuple(c2), position, self.radius)
            pygame.draw.circle(screen, tuple(c1), position, self.radius * 0.9)

            # Only draw IDs if font is provided - optimization
            if font:
                # Choose a contrasting color (black or white) based on background color brightness
                brightness = sum(c1) / 3
                text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)

                # Render the ID text
                id_text = font.render(f"{self.id}", 1, text_color)

                # Center the text on the particle
                text_pos = (
                    position[0] - id_text.get_width() // 2,
                    position[1] - id_text.get_height() // 2,
                )
                screen.blit(id_text, text_pos)

    def kill(self, space):
        space.remove(self.body, self)
        self.alive = False

    @property
    def state(self):
        return FruitState(self.body.position, self.radius, self.type)

    @property
    def pos(self):
        return np.array(self.body.position)


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
