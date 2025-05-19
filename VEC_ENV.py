import time
import argparse
import gymnasium as gym
from PIL import Image
from suika_gym import GAME_IDS, register_envs
from stable_baselines3.common.vec_env import SubprocVecEnv
from tqdm import tqdm
import numpy as np

# register_envs()
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run Suika game with specified learning level.')
    parser.add_argument('--level', type=int, default=1, choices=[1, 2, 3, 4],
                        help='Learning level (1-4) (default: 1)')
    parser.add_argument('--fps', type=int, default=120,
                        help='Frames per second for physics simulation (default: 120)')
    parser.add_argument('--render', action='store_true', default=False, 
                        help='Enable rendering')
    parser.add_argument('--render_fps', type=int, default=60,
                        help='Frames per second for rendering with pygame (default: 60)')
    parser.add_argument('--save_gif', action='store_true',
                        help='Save frames as GIF for levels 3 and 4')
    args = parser.parse_args()

    # Create environment based on level
    game_id = GAME_IDS[args.level]
    render_mode = "human" if args.render else None

    N = 16
    vec_env = SubprocVecEnv([
        lambda: gym.make(game_id, render_mode=None, render_fps=args.render_fps, fps=args.fps)
        for _ in range(N)
    ])

    current_time = time.time()
    terminated = False
    episode = 100
    i = 0
    progress_bar = tqdm(total=episode, desc="Training Progress")

    obs, info = vec_env.reset()

    while i < episode:

        action = np.array([vec_env.action_space.sample() for _ in range(N)])

        obs, reward, terminated, info = vec_env.step(action)

        for env_idx, done in enumerate(terminated):
            if done:
                i += 1
                progress_bar.update(1)


    print(f"Elapsed time: {(time.time() - current_time):.2f} seconds")

    vec_env.close()