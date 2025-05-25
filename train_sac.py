import os
import sys
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import cv2

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from suika_gym import SuikaEnv

class ListObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper to convert the problematic Sequence observation space to a fixed-size Box space
    for levels 1 and 2 (coordinate-based observations).
    """
    def __init__(self, env):
        super().__init__(env)
        
        # Define maximum number of fruits that can be on screen
        self.max_fruits = 100  # Reasonable upper bound
        
        # Create new observation space
        self.observation_space = spaces.Dict({
            # Each fruit: [x, y, radius, type] - 4 values per fruit
            # Pad with zeros for unused slots
            "boards": spaces.Box(
                low=0, 
                high=1, 
                shape=(env.n_frames, self.max_fruits * 4), 
                dtype=np.float32
            ),
            "next_fruit": spaces.Discrete(5),
        })
    
    def observation(self, obs):
        """Convert list-based observation to fixed-size array"""
        boards = obs["boards"]
        next_fruit = obs["next_fruit"]
        
        # Convert each frame's list to fixed-size array
        processed_boards = []
        for frame_fruits in boards:
            # Create array for this frame
            frame_array = np.zeros(self.max_fruits * 4, dtype=np.float32)
            
            # Fill with fruit data (limit to max_fruits)
            for i, fruit in enumerate(frame_fruits[:self.max_fruits]):
                base_idx = i * 4
                # Normalize positions to [0, 1]
                frame_array[base_idx] = fruit.pos[0] / 570  # WIDTH
                frame_array[base_idx + 1] = fruit.pos[1] / 770  # HEIGHT
                frame_array[base_idx + 2] = fruit.radius / 135  # Max radius
                frame_array[base_idx + 3] = fruit.type / 10  # Max type
            
            processed_boards.append(frame_array)
        
        return {
            "boards": np.array(processed_boards, dtype=np.float32),
            "next_fruit": next_fruit
        }

class ImageObservationWrapper(gym.ObservationWrapper):
    """
    Memory-efficient wrapper for image-based observations.
    Reduces image size and converts to grayscale to save memory.
    """
    def __init__(self, env, img_size=(84, 84), grayscale=True):
        super().__init__(env)
        self.img_size = img_size
        self.grayscale = grayscale
        
        # Calculate channels
        channels = 1 if grayscale else 3
        
        # Create new observation space with reduced size
        self.observation_space = spaces.Dict({
            "boards": spaces.Box(
                low=0,
                high=255,
                shape=(env.n_frames, img_size[0], img_size[1], channels),
                dtype=np.uint8  # Keep as uint8 to save memory
            ),
            "next_fruit": spaces.Discrete(5),
        })
    
    def observation(self, obs):
        """Resize and optionally convert images to grayscale"""
        boards = obs["boards"]
        next_fruit = obs["next_fruit"]
        
        processed_boards = []
        for frame in boards:
            # Resize image
            resized = cv2.resize(frame, self.img_size, interpolation=cv2.INTER_AREA)
            
            # Convert to grayscale if requested
            if self.grayscale and len(resized.shape) == 3:
                resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
                resized = np.expand_dims(resized, axis=-1)
            
            processed_boards.append(resized)
        
        return {
            "boards": np.array(processed_boards, dtype=np.uint8),
            "next_fruit": next_fruit
        }

def make_env(level=1, n_frames=8, rank=0, seed=0, img_size=(84, 84), grayscale=True):
    """
    Utility function for multiprocessed env.
    
    Args:
        level: Game level (1-4)
        n_frames: Number of frames to capture
        rank: Index of the subprocess
        seed: Random seed
        img_size: Target image size for resizing (width, height)
        grayscale: Whether to convert images to grayscale
    """
    def _init():
        env = SuikaEnv(level=level, n_frames=n_frames, render_mode=None)
        
        # Apply appropriate wrapper based on level
        if level in [1, 2]:
            # Coordinate-based observations
            env = ListObservationWrapper(env)
        else:
            # Image-based observations - apply memory-efficient wrapper
            env = ImageObservationWrapper(env, img_size=img_size, grayscale=grayscale)
        
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def train_sac_agent(
    level=1,
    n_frames=8,
    total_timesteps=1000000,
    n_envs=4,
    learning_rate=3e-4,
    buffer_size=50000,  # Reduced buffer size for memory efficiency
    batch_size=128,     # Reduced batch size
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    target_update_interval=1,
    learning_starts=5000,  # Reduced learning starts
    use_sde=False,
    sde_sample_freq=-1,
    use_sde_at_warmup=False,
    model_save_path="models/sac_suika",
    log_dir="logs/sac_suika",
    eval_freq=10000,
    n_eval_episodes=5,
    save_freq=25000,
    seed=42,
    img_size=(84, 84),    # Smaller image size
    grayscale=True        # Use grayscale to reduce memory
):
    """
    Train a SAC agent on the Suika game environment with memory optimizations.
    
    Args:
        level: Game level (1-4, see SuikaEnv documentation)
        n_frames: Number of frames to capture per step
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        learning_rate: Learning rate for the optimizer
        buffer_size: Size of the replay buffer (reduced for memory efficiency)
        batch_size: Batch size for training (reduced for memory efficiency)
        tau: Soft update coefficient for target networks
        gamma: Discount factor
        train_freq: Training frequency
        gradient_steps: Number of gradient steps per update
        target_update_interval: Target network update interval
        learning_starts: Number of steps before training starts
        use_sde: Whether to use State Dependent Exploration
        sde_sample_freq: Sample frequency for SDE
        use_sde_at_warmup: Whether to use SDE during warmup
        model_save_path: Path to save the trained model
        log_dir: Directory for tensorboard logs
        eval_freq: Evaluation frequency
        n_eval_episodes: Number of episodes for evaluation
        save_freq: Model checkpoint save frequency
        seed: Random seed
        img_size: Target image size for resizing (width, height)
        grayscale: Whether to convert images to grayscale
    """
    
    # Create directories
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Training SAC agent on Suika Game Level {level}")
    print(f"Using {n_envs} parallel environments")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Buffer size: {buffer_size} (reduced for memory efficiency)")
    if level > 2:
        print(f"Image size: {img_size}, Grayscale: {grayscale}")
    
    # Create vectorized training environment
    if n_envs == 1:
        # Single environment
        env = DummyVecEnv([make_env(
            level=level, n_frames=n_frames, rank=0, seed=seed,
            img_size=img_size, grayscale=grayscale
        )])
    else:
        # Multiple environments with multiprocessing
        env = SubprocVecEnv([
            make_env(
                level=level, n_frames=n_frames, rank=i, seed=seed,
                img_size=img_size, grayscale=grayscale
            )
            for i in range(n_envs)
        ])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(
        level=level, n_frames=n_frames, rank=0, seed=seed + 1000,
        img_size=img_size, grayscale=grayscale
    )])
    
    # Define SAC policy based on level
    if level in [1, 2]:
        # For coordinate-based observations, use MLP policy
        policy = "MultiInputPolicy"
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], qf=[256, 256]),
            activation_fn=torch.nn.ReLU,
        )
    else:
        # For image-based observations, use CNN policy
        policy = "MultiInputPolicy"
        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(
                features_dim=256,
                img_size=img_size,
                grayscale=grayscale
            ),
            net_arch=dict(pi=[256, 256], qf=[256, 256]),
        )
    
    # Create SAC model
    model = SAC(
        policy,
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        target_update_interval=target_update_interval,
        learning_starts=learning_starts,
        use_sde=use_sde,
        sde_sample_freq=sde_sample_freq,
        use_sde_at_warmup=use_sde_at_warmup,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        verbose=1,
        seed=seed,
        device="auto"
    )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{model_save_path}_best",
        log_path=log_dir,
        eval_freq=eval_freq // n_envs,  # Adjust for vectorized envs
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,  # Adjust for vectorized envs
        save_path=f"{model_save_path}_checkpoints/",
        name_prefix="sac_suika"
    )
    
    callback_list = CallbackList([eval_callback, checkpoint_callback])
    
    # Train the model
    print("Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        tb_log_name=f"SAC_level_{level}"
    )
    
    # Save final model
    model.save(f"{model_save_path}_final")
    print(f"Training completed! Model saved to {model_save_path}_final")
    
    # Clean up
    env.close()
    eval_env.close()
    
    return model

# Custom CNN feature extractor for image-based observations
class CustomCNN(BaseFeaturesExtractor):
    """
    Memory-efficient CNN feature extractor for Suika game images.
    """
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256, 
                 img_size=(84, 84), grayscale=True):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        
        self.img_size = img_size
        self.grayscale = grayscale
        
        # Get the shape of the boards (image frames)
        boards_shape = observation_space['boards'].shape
        n_frames = boards_shape[0]
        channels_per_frame = boards_shape[-1]
        n_input_channels = n_frames * channels_per_frame
        
        # More memory-efficient CNN architecture
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample_input = torch.zeros(1, n_input_channels, img_size[0], img_size[1])
            n_flatten = self.cnn(sample_input).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + 5, features_dim),  # +5 for next_fruit one-hot
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Extract boards and next_fruit
        boards = observations['boards']
        next_fruit = observations['next_fruit']
        
        # Debug prints to understand shapes
        # print(f"Raw next_fruit.shape: {next_fruit.shape}")
        # print(f"Raw boards.shape: {boards.shape}")
        
        # Convert uint8 to float32 and normalize to [0, 1]
        boards = boards.float() / 255.0
        
        # Reshape boards: (batch, frames, height, width, channels) -> (batch, frames*channels, height, width)
        batch_size = boards.shape[0]
        boards = boards.permute(0, 1, 4, 2, 3)  # (batch, frames, channels, height, width)
        boards = boards.reshape(batch_size, -1, boards.shape[-2], boards.shape[-1])
        
        # Process through CNN
        cnn_features = self.cnn(boards)
        
        # Fix next_fruit shape - flatten and ensure it's 1D per batch
        if len(next_fruit.shape) > 1:
            # If next_fruit has extra dimensions, take the first element or flatten
            next_fruit = next_fruit.flatten()
            if len(next_fruit) != batch_size:
                # If we have too many elements, take first batch_size elements
                next_fruit = next_fruit[:batch_size]
        
        # Ensure next_fruit is the right shape [batch_size]
        next_fruit = next_fruit.reshape(batch_size)
        
        # print(f"Fixed next_fruit.shape before one-hot: {next_fruit.shape}")
        
        # One-hot encode next_fruit
        next_fruit_onehot = torch.nn.functional.one_hot(next_fruit.long(), num_classes=5).float()
        
        # print(f"cnn_features.shape: {cnn_features.shape}")
        # print(f"next_fruit_onehot.shape: {next_fruit_onehot.shape}")
        
        # Concatenate features
        combined_features = torch.cat([cnn_features, next_fruit_onehot], dim=1)
        
        return self.linear(combined_features)

if __name__ == "__main__":
    # Level 1: Coordinate-size list with game engine access
    # print("Training on Level 1 (Coordinate-based with engine access)")
    # train_sac_agent(
    #     level=1,
    #     n_frames=8,  # Fewer frames for coordinate-based
    #     total_timesteps=5000000,
    #     n_envs=8,
    #     model_save_path="models/sac_suika_level1",
    #     log_dir="logs/sac_suika_level1"
    # )
    
    # Level 3: Image with game engine access (memory optimized)
    print("Training on Level 3 (Image-based with engine access)")
    train_sac_agent(
        level=3,
        n_frames=4,  # Reduced frames to save memory
        total_timesteps=5000000,
        n_envs=4,    # Reduced parallel environments
        buffer_size=20000,  # Much smaller buffer
        batch_size=64,      # Smaller batch size
        learning_starts=2500,
        model_save_path="models/sac_suika_level3",
        log_dir="logs/sac_suika_level3",
        img_size=(84 * 2, 84 * 2),  # Smaller image size
        grayscale=False      # Use grayscale
    )