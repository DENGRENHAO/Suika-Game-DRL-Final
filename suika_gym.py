import sys
import numpy as np
import pygame
import pymunk
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

SIZE = WIDTH, HEIGHT = np.array([570, 770])
PAD = (24, 160)
A = (PAD[0], PAD[1])
B = (PAD[0], HEIGHT - PAD[0])
C = (WIDTH - PAD[0], HEIGHT - PAD[0])
D = (WIDTH - PAD[0], PAD[1])
BG_COLOR = (250, 240, 148)
W_COLOR = (250, 190, 58)
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
THICKNESS = 14
DENSITY = 0.001
ELASTICITY = 0.1
IMPULSE = 10000
GRAVITY = 2000
DAMPING = 0.8
BIAS = 0.00001
POINTS = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66]

# Environment IDs for the four different learning levels
GAME_IDS = {
    1: "suika-game-l1-v0",  # Coordinate-size list with game engine access
    2: "suika-game-l2-v0",  # Coordinate-size list without game engine access
    3: "suika-game-l3-v0",  # Image with game engine access
    4: "suika-game-l4-v0",  # Image without game engine access
}

class SuikaEnv(gym.Env):
    def __init__(self, render_mode=None, render_fps=60, fps=120, level=1):
        """
        Initialize the Suika game environment
        
        Args:
            render_mode: "human" for window display
            fps: Frames per second for physics simulation
            level: Learning level (1-4)
                   1: Coordinate-size list with game engine access
                   2: Coordinate-size list without game engine access
                   3: Image with game engine access
                   4: Image without game engine access
        """
        self.SIZE = SIZE
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.PAD = PAD
        self.A = A
        self.B = B
        self.C = C 
        self.D = D
        self.BG_COLOR = BG_COLOR
        self.W_COLOR = W_COLOR
        self.COLORS = COLORS
        self.FPS = fps
        self.RADII = RADII
        self.THICKNESS = THICKNESS
        self.DENSITY = DENSITY
        self.ELASTICITY = ELASTICITY
        self.IMPULSE = IMPULSE
        self.GRAVITY = GRAVITY
        self.DAMPING = DAMPING
        self.NEXT_DELAY = self.FPS
        self.BIAS = BIAS
        self.POINTS = POINTS
        self.render_fps = render_fps
        self.fast_mode = False
        self.level = level
        
        # Validate level
        if self.level not in [1, 2, 3, 4]:
            raise ValueError(f"Invalid level: {self.level}. Must be 1, 2, 3, or 4.")
        
        self.cleaup_interval = 40
        self.render_interval = 4
        self.max_stability_steps = 20
        
        # Grid size for coordinate representation
        self.grid_size = (WIDTH // 10, HEIGHT // 10)
        
        # Image size for image-based representation (levels 3 and 4)
        self.image_size = (WIDTH, HEIGHT)
        
        # Unique ID counter for particles
        self.next_particle_id = 1
        
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
        self.shape_to_particle = {}
        self.particles = []
        self.score = 0
        self.steps_since_drop = 0
        self.game_over = False
        self.current_step = 0
        
        # Setup pymunk space
        self._setup_space()
        
        # Initialize next particle
        self.next_particle_type = self.rng.integers(0, 5)
        self.next_particle_x = self.WIDTH // 2
    
    def _setup_observation_space(self):
        """Set up observation space based on level"""
        # If levels 1 or 2 (coordinate-size list)
        if self.level in [1, 2]:
            self.observation_space = spaces.Dict({
                "grid": spaces.Box(low=0, high=11, shape=(self.grid_size[0], self.grid_size[1]), dtype=np.int8),
                "next_fruit": spaces.Discrete(5)  # 0-4 for the next fruit types
            })
        # If levels 3 or 4 (image-based)
        else:
            self.observation_space = spaces.Dict({
                "image": spaces.Box(low=0, high=255, shape=(self.image_size[1], self.image_size[0], 3), dtype=np.uint8),
                "next_fruit": spaces.Discrete(5)
            })
        
    def _setup_space(self):
        """Set up the pymunk physics space and walls"""
        self.space = pymunk.Space()
        self.space.gravity = (0, self.GRAVITY)
        self.space.damping = self.DAMPING
        self.space.collision_bias = self.BIAS
        
        # Create walls
        self.walls = []
        self.walls.append(self._create_wall(self.A, self.B))  # left
        self.walls.append(self._create_wall(self.B, self.C))  # bottom
        self.walls.append(self._create_wall(self.C, self.D))  # right
        
        # Set up collision handler
        handler = self.space.add_collision_handler(1, 1)
        handler.begin = self._collide
        # Store collision data directly in the environment since we can't set handler.data
        self.collision_data = {"mapper": self.shape_to_particle, 
                              "particles": self.particles, 
                              "score": 0}
    
    def _create_wall(self, a, b):
        """Create a wall segment in the physics space"""
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        shape = pymunk.Segment(body, a, b, self.THICKNESS // 2)
        shape.friction = 10
        self.space.add(body, shape)
        return (body, shape)
    
    def _init_render(self):
        """Initialize rendering components"""
        if not self._pygame_initialized:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
                pygame.display.set_caption(f"Suika Gym - Level {self.level}")
            else:
                self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
            
            self.clock = pygame.time.Clock()
            pygame.font.init()
            self.scorefont = pygame.font.SysFont("monospace", 32)
            self.overfont = pygame.font.SysFont("monospace", 72)
            self.idfont = pygame.font.SysFont("monospace", 12)
            self._pygame_initialized = True
    
    def _collide(self, arbiter, space, data):
        """Collision handler for fruits"""
        sh1, sh2 = arbiter.shapes
        _mapper = self.collision_data["mapper"]
        if sh1 not in _mapper or sh2 not in _mapper:
            return True
            
        pa1 = _mapper[sh1]
        pa2 = _mapper[sh2]
        cond = bool(pa1.n != pa2.n)
        pa1.has_collided = cond
        pa2.has_collided = cond
        
        if not cond:  # Same fruit type, merge them
            new_particle = self._resolve_collision(pa1, pa2, space, self.collision_data["particles"], _mapper)
            if new_particle:
                self.collision_data["particles"].append(new_particle)
                self.collision_data["score"] += self.POINTS[pa1.n]
                self.score = self.collision_data["score"]  # Update the environment score
        
        return cond  # Return True to allow the collision, False to ignore it
    
    def _resolve_collision(self, p1, p2, space, particles, mapper):
        """Resolve collision between two identical fruits"""
        if p1.n == p2.n:
            distance = np.linalg.norm(p1.pos - p2.pos)
            if distance < 2 * p1.radius:
                p1.kill(space)
                p2.kill(space)
                
                # Create new merged particle
                new_pos = np.mean([p1.pos, p2.pos], axis=0)
                pn = Particle(new_pos, p1.n+1, space, mapper, self.next_particle_id)
                self.next_particle_id += 1
                
                # Apply impulses to nearby particles
                for p in particles:
                    if p.alive:
                        vector = p.pos - pn.pos
                        distance = np.linalg.norm(vector)
                        if distance < pn.radius + p.radius:
                            impulse = self.IMPULSE * vector / (distance ** 2)
                            p.body.apply_impulse_at_local_point(tuple(impulse))
                
                return pn
        return None
    
    def _get_grid_observation(self):
        """Get grid-based observation (for levels 1 and 2)"""
        # Create empty grid
        grid = np.zeros(self.grid_size, dtype=np.int8)
        
        # Map particles to grid
        scale_x = self.grid_size[0] / self.WIDTH
        scale_y = self.grid_size[1] / self.HEIGHT
        
        for p in self.particles:
            if p.alive:
                try:
                    # Check if position contains NaN values
                    if np.isnan(p.pos).any():
                        print(f"Warning: Particle ID {p.id} has NaN position: {p.pos}, type: {p.n}, alive: {p.alive}")
                        continue
                    
                    # Convert particle position to grid coordinates
                    x, y = int(p.pos[0] * scale_x), int(p.pos[1] * scale_y)
                    
                    # Calculate radius in grid units
                    radius = int(p.radius * scale_x)
                    
                    # Add particle to grid, handling bounds
                    x_min = max(0, x - radius)
                    x_max = min(self.grid_size[0], x + radius + 1)
                    y_min = max(0, y - radius)
                    y_max = min(self.grid_size[1], y + radius + 1)
                    
                    # Fill circle area with particle type value
                    for gx in range(x_min, x_max):
                        for gy in range(y_min, y_max):
                            if ((gx - x)**2 + (gy - y)**2) <= radius**2:
                                grid[gx, gy] = p.n + 1  # +1 so empty space is 0
                except (ValueError, TypeError) as e:
                    print(f"Warning: Error processing particle ID {p.id}: {e}")
                    continue
        
        return grid
    
    def _get_image_observation(self):
        """Get image-based observation (for levels 3 and 4)"""
        # Ensure render component is initialized
        if not hasattr(self, 'screen'):
            if not self._pygame_initialized:
                # Initialize a hidden screen for rendering
                pygame.init()
                self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
                pygame.font.init()
                self.idfont = None  # Don't show IDs in the observation
                self._pygame_initialized = True
        
        # Render the game state to the surface
        self.screen.fill(self.BG_COLOR)
        
        # Draw next particle indicator
        if not self.game_over:
            self._draw_next_particle()
        
        # Draw walls
        for wall in self.walls:
            pygame.draw.line(self.screen, self.W_COLOR, 
                            wall[1].a, wall[1].b, self.THICKNESS)
        
        # Draw particles
        active_particles = [p for p in self.particles if p.alive]
        for p in active_particles:
            p.draw(self.screen, None)  # Don't draw IDs in the observation
        
        # Scale the surface to the desired size
        scaled_surface = pygame.transform.scale(self.screen, self.image_size)
        
        # Convert to numpy array
        img_array = np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_surface)), axes=(1, 0, 2)
        )
        
        return img_array
    
    def _get_observation(self):
        """Get observation based on level"""
        if self.level in [1, 2]:
            # Grid-based observation
            return {
                "grid": self._get_grid_observation(),
                "next_fruit": self.next_particle_type
            }
        else:
            # Image-based observation
            return {
                "image": self._get_image_observation(),
                "next_fruit": self.next_particle_type
            }
    
    def _get_engine_state(self):
        """Get the internal game engine state (positions and radii of particles)"""
        engine_state = []
        for p in self.particles:
            if p.alive:
                # Only include non-NaN positions
                if not np.isnan(p.pos).any():
                    engine_state.append({
                        "id": p.id,
                        "position": p.pos.tolist(),
                        "radius": p.radius,
                        "type": p.n
                    })
        return engine_state
    
    def check_game_over(self):
        """Check if game is over (fruit above threshold line)"""
        for p in self.particles:
            if p.alive and p.has_collided and p.pos[1] < self.PAD[1]:
                return True
        return False

    def _clean_invalid_particles(self):
        """Clean up particles with invalid positions"""
        to_remove = []
        for p in self.particles:
            if p.alive and np.isnan(p.pos).any():
                p.kill(self.space)
                to_remove.append(p)
        
        # Remove killed particles from the list
        for p in to_remove:
            try:
                self.particles.remove(p)
            except ValueError:
                pass
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Clear old state
        self.space = None
        self.particles = []
        self.shape_to_particle = {}
        self.score = 0
        self.steps_since_drop = 0
        self.game_over = False
        self.current_step = 0
        
        # Reset particle ID counter
        self.next_particle_id = 1
        
        # Re-initialize the physics space
        self._setup_space()
        
        # Generate new first fruit
        self.next_particle_type = self.rng.integers(0, 5)
        self.next_particle_x = self.WIDTH // 2
        
        # Get observation
        observation = self._get_observation()
        
        # Prepare info dict based on level
        info = {"score": self.score}
        
        # Add engine state for levels 1 and 3
        if self.level in [1, 3]:
            info["engine_state"] = self._get_engine_state()
        
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
        x_min = self.PAD[0] + self.RADII[self.next_particle_type] + self.THICKNESS // 2
        x_max = self.WIDTH - x_min
        x_pos = x_min + action[0] * (x_max - x_min)
        
        # Create and drop new particle
        old_score = self.score
        new_particle = Particle((x_pos, self.PAD[1] // 2), 
                              self.next_particle_type, 
                              self.space, 
                              self.shape_to_particle,
                              self.next_particle_id)
        self.next_particle_id += 1
        self.particles.append(new_particle)
        
        # Run physics for a fixed amount of time
        stable = False
        stability_counter = 0
        
        # Run physics until stable or max steps
        for i in range(self.FPS):
            # Periodically clean up invalid particles
            if i % self.cleaup_interval == 0:
                self._clean_invalid_particles()
            
            self.space.step(1/self.FPS)
            
            # Render during physics simulation if render_mode is set
            if self.render_mode is not None and i % self.render_interval == 0:
                self._render_frame()
                
                if self.render_mode == "human":
                    # Process events to keep the window responsive
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
            
            # Check if particles are stable (only check every few steps)
            all_stable = True
            # Only check a subset of particles for performance
            active_particles = [p for p in self.particles if p.alive]
            
            for p in active_particles:
                if abs(p.body.velocity[0]) > 10 or abs(p.body.velocity[1]) > 10:
                    all_stable = False
                    break
            
            if all_stable:
                stability_counter += 1
                if stability_counter >= self.max_stability_steps:
                    stable = True
                    break
            else:
                stability_counter = 0
                
            if self.check_game_over():
                self.game_over = True
                break
        
        self._clean_invalid_particles()
        
        # Generate next fruit
        self.next_particle_type = self.rng.integers(0, 5)
        
        # Calculate reward
        reward = self.score - old_score  # Reward based on points gained
        
        # Check termination conditions
        terminated = self.game_over
        truncated = False
        
        # Get observation
        observation = self._get_observation()
        
        # Prepare info dict based on level
        info = {
            "score": self.score,
            "stable": stable
        }
        
        # Add engine state for levels 1 and 3
        if self.level in [1, 3]:
            info["engine_state"] = self._get_engine_state()
        
        return observation, reward, terminated, truncated, info
    
    def _draw_next_particle(self):
        """Draw the next particle indicator"""
        n = self.next_particle_type
        radius = self.RADII[n]
        c1 = np.array(self.COLORS[n])
        c2 = (c1 * 0.8).astype(int)
        pygame.draw.circle(self.screen, tuple(c2), 
                          (self.next_particle_x, self.PAD[1] // 2), radius)
        pygame.draw.circle(self.screen, tuple(c1), 
                          (self.next_particle_x, self.PAD[1] // 2), radius * 0.9)
    
    def render(self):
        """Render the environment"""
        if self.render_mode is None:
            return
        
        return self._render_frame()

    def _render_frame(self):
        """Render a single frame of the environment"""
        if not hasattr(self, 'screen'):
            return None
            
        # Fill background
        self.screen.fill(self.BG_COLOR)
        
        # Draw next particle indicator
        if not self.game_over:
            self._draw_next_particle()
        
        # Draw walls
        for wall in self.walls:
            pygame.draw.line(self.screen, self.W_COLOR, 
                            wall[1].a, wall[1].b, self.THICKNESS)
        
        # Draw particles - optimization: only draw active ones
        active_particles = [p for p in self.particles if p.alive]
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
            self.screen.blit(game_over_label, self.PAD)
        
        # Display or return the screen
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.render_fps)
        
        # if self.render_mode == "rgb_array":
        #     return np.transpose(
        #         np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        #     )
        
        return None
    
    def close(self):
        """Clean up resources"""
        if self._pygame_initialized and self.render_mode == "human":
            self._pygame_initialized = False
            pygame.quit()
            sys.exit()


class Particle:
    """Represents a fruit in the Suika game"""
    
    def __init__(self, pos, n, space, mapper, particle_id):
        self.id = particle_id  # Unique ID for each particle
        self.n = n % 11
        self.radius = RADII[self.n]  # Reference the global constants directly
        self.body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        self.body.position = tuple(pos)
        self.shape = pymunk.Circle(body=self.body, radius=self.radius)
        self.shape.density = DENSITY
        self.shape.elasticity = ELASTICITY
        self.shape.collision_type = 1
        self.shape.friction = 0.2
        self.has_collided = False
        mapper[self.shape] = self
        
        space.add(self.body, self.shape)
        self.alive = True
    
    def draw(self, screen, font):
        if self.alive:
            c1 = np.array(COLORS[self.n])
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
                text_pos = (position[0] - id_text.get_width() // 2, 
                           position[1] - id_text.get_height() // 2)
                screen.blit(id_text, text_pos)
    
    def kill(self, space):
        space.remove(self.body, self.shape)
        self.alive = False
    
    @property
    def pos(self):
        return np.array(self.body.position)


# Register the environments with Gymnasium
def register_envs():
    for level, game_id in GAME_IDS.items():
        try:
            register(
                id=game_id,
                entry_point='suika_gym:SuikaEnv',
                kwargs={'level': level}
            )
        except Exception as e:
            print(f"Registration note for level {level}: {e}")

register_envs()


# Example usage
if __name__ == "__main__":
    import time
    import argparse
    from PIL import Image
    
    parser = argparse.ArgumentParser(description='Run Suika game with specified learning level.')
    parser.add_argument('--level', type=int, default=1, choices=[1, 2, 3, 4],
                        help='Learning level (1-4) (default: 1)')
    parser.add_argument('--fps', type=int, default=120,
                        help='Frames per second for physics simulation (default: 120)')
    parser.add_argument('--render', action='store_true',
                        help='Enable rendering')
    parser.add_argument('--render_fps', type=int, default=60,
                        help='Frames per second for rendering with pygame (default: 60)')
    parser.add_argument('--save_gif', action='store_true',
                        help='Save frames as GIF for levels 3 and 4')
    args = parser.parse_args()
    
    # Create environment based on level
    game_id = GAME_IDS[args.level]
    render_mode = "human" if args.render else None
    
    env = gym.make(game_id, render_mode=render_mode, render_fps=args.render_fps, fps=args.fps)
    print(f"Created environment with level {args.level}: {game_id}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    current_time = time.time()
    for i in range(3):
        obs, info = env.reset()
        
        step_count = 0
        frames = []
        while step_count < 20:
            action = env.action_space.sample()
            obs, reward, terminated, _, info = env.step(action)
            
            if args.level in [1, 2]:
                print(f"Grid shape: {obs['grid'].shape}, Next fruit: {obs['next_fruit']}")
            else:
                print(f"Image shape: {obs['image'].shape}, Next fruit: {obs['next_fruit']}")
                
                if args.save_gif:
                    frame = Image.fromarray(obs['image'])
                    frames.append(frame)
                
            if 'engine_state' in info:
                print(f"Engine state available: {len(info['engine_state'])} particles")
                for particle in info['engine_state']:
                    print(f"Particle ID: {particle['id']}, Position: {particle['position']}, Radius: {particle['radius']}, Type: {particle['type']}")
            
            print(f"Step {step_count}, Score: {info['score']}, Reward: {reward}")
            step_count += 1
            
            if terminated:
                print(f"Game {i+1} over! Score: {info['score']}")
                break
        
        if args.level in [3, 4] and args.save_gif:
            frames[0].save(f"game_{i+1}.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
    
    print(f"Elapsed time: {(time.time() - current_time):.2f} seconds")
    
    env.close()