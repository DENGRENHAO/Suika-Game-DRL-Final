import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import sys
import os

from agents.base_agent import Agent

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from suika_gym import SuikaEnv
from wrappers import CoordSizeToImage

env = CoordSizeToImage(SuikaEnv(level=1, n_frames=1))
# Hyperparameters (some from original, some new/modified)
HIDDEN_SIZE = 512 # Hidden size for FC layers after feature extraction
BUFFER_SIZE = 10000
LR = 3e-4
GAMMA = 0.99
TAU = 0.005
BATCH_SIZE = 256

IMAGE_CHANNELS = 1  # Grayscale image
IMAGE_HEIGHT = 96   # Resized image height
IMAGE_WIDTH = 96    # Resized image width
NUM_FRUIT_TYPES = 11 # Example: Number of unique fruit types in Suika
FRUIT_EMBEDDING_DIM = 16 # Example: Embedding dimension for fruit type
ACTION_SIZE = 1     # Single dimension for horizontal position

from collections import deque
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tqdm import tqdm

NUM_EPISODES = 10000
WARMUP_EPISODES = 50

class Actor(nn.Module):
    def __init__(self, image_shape, num_fruit_types, fruit_embedding_dim, action_size, hidden_size):
        super().__init__()
        C, H, W = image_shape # C will be 1 for grayscale
        self.action_size = action_size

        # Image processing backbone
        self.conv_base = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4), # Input channels C (now 1)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten() 
        )
        # Calculate CNN output size
        with torch.no_grad():
            dummy_img = torch.zeros(1, C, H, W)
            cnn_out_dim = self.conv_base(dummy_img).shape[1]

        # Fruit type processing
        self.fruit_embed = nn.Embedding(num_fruit_types, fruit_embedding_dim)

        # Combined feature processing
        combined_features_dim = cnn_out_dim + fruit_embedding_dim
        self.fc_backbone = nn.Sequential(
            nn.Linear(combined_features_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        self.mean_layer = nn.Linear(hidden_size, action_size)
        self.log_std_layer = nn.Linear(hidden_size, action_size)

    def forward(self, image_state, fruit_state):
        # image_state: (batch, IMAGE_CHANNELS, H, W)
        # fruit_state: (batch,) LongTensor for embedding
        
        img_features = self.conv_base(image_state)
        
        # fruit_state should already be (batch,) and LongTensor
        fruit_features = self.fruit_embed(fruit_state) # (batch, fruit_embedding_dim)

        combined_features = torch.cat([img_features, fruit_features], dim=1)
        
        x = self.fc_backbone(combined_features)
        
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2) 
        std = torch.exp(log_std)
        
        return mean, std

    def sample(self, image_state, fruit_state):
        mean, std = self.forward(image_state, fruit_state)
        normal = Normal(mean, std)
        x_t = normal.rsample()  
        action_tanh = torch.tanh(x_t) 
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.sum(torch.log(1 - action_tanh.pow(2) + 1e-6), dim=-1, keepdim=True)

        # Ensure log_prob is (batch, 1)
        if log_prob.ndim > 1 and log_prob.shape[-1] != 1 and self.action_size == 1:
             log_prob = log_prob.sum(1, keepdim=True)
        elif log_prob.ndim == 1 and self.action_size == 1:
            log_prob = log_prob.unsqueeze(-1)

        return action_tanh, log_prob

class Critic(nn.Module):
    def __init__(self, image_shape, num_fruit_types, fruit_embedding_dim, action_size, hidden_size):
        super().__init__()
        C, H, W = image_shape # C will be 1 for grayscale

        self.conv_base = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4), # Input channels C (now 1)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy_img = torch.zeros(1, C, H, W)
            cnn_out_dim = self.conv_base(dummy_img).shape[1]

        self.fruit_embed = nn.Embedding(num_fruit_types, fruit_embedding_dim)

        combined_features_dim = cnn_out_dim + fruit_embedding_dim + action_size

        self.q1 = nn.Sequential(
            nn.Linear(combined_features_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.q2 = nn.Sequential(
            nn.Linear(combined_features_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, image_state, fruit_state, action):
        img_features = self.conv_base(image_state)
        
        # fruit_state should already be (batch,) and LongTensor
        fruit_features = self.fruit_embed(fruit_state)
        
        x = torch.cat([img_features, fruit_features, action], dim=1)
        
        return self.q1(x), self.q2(x)

class SACAgent(Agent):
    def __init__(self, image_shape=(IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), 
                 num_fruit_types=NUM_FRUIT_TYPES, 
                 fruit_embedding_dim=FRUIT_EMBEDDING_DIM,
                 action_size=ACTION_SIZE,
                 hidden_size=HIDDEN_SIZE,
                 lr=LR, gamma=GAMMA, tau=TAU, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
                 alpha_lr=None): # Added alpha_lr
        
        self.action_size = action_size
        self.batch_size = batch_size # Store batch_size
        self.actor = Actor(image_shape, num_fruit_types, fruit_embedding_dim, action_size, hidden_size).to(device)
        self.critic = Critic(image_shape, num_fruit_types, fruit_embedding_dim, action_size, hidden_size).to(device)
        self.target_critic = Critic(image_shape, num_fruit_types, fruit_embedding_dim, action_size, hidden_size).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for param in self.target_critic.parameters(): 
            param.requires_grad = False

        self.gamma = gamma
        self.tau = tau
        
        self.target_entropy = torch.tensor(-action_size, dtype=torch.float32, device=device)
        self.log_alpha = torch.tensor(np.log(0.2), requires_grad=True, device=device) 
        
        if alpha_lr is None: # If alpha_lr is not specified, use the main lr
            alpha_lr = lr
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr) 
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(BUFFER_SIZE, device=device), 
            batch_size=BATCH_SIZE
        )

        self.load("models/sac_agent")

    def update(self):
        # Check if buffer has enough samples
        batch = self.replay_buffer.sample()
        image_state = batch["state"].float() / 255.0
        fruit_state = batch["cur_fruit"] # Expected (batch_size,) LongTensor
        action = batch["action"]
        reward = batch["reward"].unsqueeze(1)
        next_image_state = batch["next_state"].float() / 255.0
        next_fruit_state = batch["next_cur_fruit"] # Expected (batch_size,) LongTensor
        done = batch["done"].unsqueeze(1)

        alpha = self.log_alpha.exp().detach() 

        with torch.no_grad():
            next_action_tanh, next_log_prob = self.actor.sample(next_image_state, next_fruit_state)
            target_q1, target_q2 = self.target_critic(next_image_state, next_fruit_state, next_action_tanh)
            target_q_min = torch.min(target_q1, target_q2)
            q_target = reward + (1 - done) * self.gamma * (target_q_min - alpha * next_log_prob)

        # print("Reward shape:", reward.shape)
        # print("Done shape:", done.shape)
        # print("Alpha shape:", alpha.shape)
        # print("Next Log Prob shape:", next_log_prob.shape)
        # print("Target Q1 shape:", target_q1.shape)
        # print("Target Q2 shape:", target_q2.shape)
        # print("Q Target shape:", q_target.shape)

        current_q1, current_q2 = self.critic(image_state, fruit_state, action)
        
        critic_loss_q1 = F.mse_loss(current_q1, q_target)
        critic_loss_q2 = F.mse_loss(current_q2, q_target)
        critic_loss = critic_loss_q1 + critic_loss_q2
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for p in self.critic.parameters():
            p.requires_grad = False

        new_action_tanh, log_prob = self.actor.sample(image_state, fruit_state)
        q1_new, q2_new = self.critic(image_state, fruit_state, new_action_tanh)
        q_new_min = torch.min(q1_new, q2_new)
        
        actor_loss = (self.log_alpha.exp() * log_prob - q_new_min).mean() # Use self.log_alpha.exp() directly
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for p in self.critic.parameters():
            p.requires_grad = True

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return actor_loss.item(), critic_loss.item(), self.log_alpha.exp().item()


    def select_action(self, obs, deterministic=True):
        obs['boards'] = obs['boards'][0]
        image_tensor, fruit_obs_id = obs['boards'], obs['cur_fruit']
        image_tensor = image_tensor.unsqueeze(0).to(device).float() / 255.0
        fruit_tensor = torch.tensor([fruit_obs_id], dtype=torch.long, device=device) # (1,) - Correct for embedding

        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(image_tensor, fruit_tensor)
                action_internal = torch.tanh(mean) 
            else:
                action_internal, _ = self.actor.sample(image_tensor, fruit_tensor) 
        
        action_scaled_0_1 = (action_internal + 1.0) / 2.0
        
        return action_scaled_0_1.cpu().numpy()[0]

    def save(self, filename_prefix):
        torch.save(self.actor.state_dict(), f"{filename_prefix}_actor.pth")
        print(f"Models and optimizers saved with prefix {filename_prefix}")

    def load(self, filename_prefix):
        self.actor.load_state_dict(torch.load(f"{filename_prefix}_actor.pth"))
        print(f"Models loaded from prefix {filename_prefix}.")



# The following block demonstrates how to run the RandomAgent.
if __name__ == "__main__":
    print("Running SACAgent example...")
    env = CoordSizeToImage(SuikaEnv(level=1))

    obs, info = env.reset()
    agent = SACAgent()

    done = False
    total_reward = 0
    episode_length = 0

    while not done:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_length += 1
        done = terminated or truncated

    print(f"SACAgent episode finished.")
    print(f"Total Reward: {total_reward}")
    print(f"Episode Length: {episode_length}")

    env.close()
