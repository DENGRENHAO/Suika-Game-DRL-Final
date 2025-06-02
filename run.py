import os
import numpy as np
import imageio
import gymnasium as gym
from suika_gym import SuikaEnv

def run_and_save_gif(
    env: gym.Env,
    num_steps: int,
    gif_path: str,
    fps: int = 30,
    seed: int = None
):
    if seed is not None:
        np.random.seed(seed)
        env.reset(seed=seed)

    obs, info = env.reset()
    frames = []

    for img in obs["boards"]:
        frames.append(img)

    for step in range(num_steps):
        action = np.random.rand(1).astype(np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        for img in obs["boards"]:
            frames.append(img)

        if terminated:
            print(f"第 {step+1} 步時遊戲結束 (terminated)。")
            break

    os.makedirs(os.path.dirname(gif_path) or ".", exist_ok=True)

    imageio.mimsave(gif_path, frames, fps=fps, loop = 0)
    print(f"GIF 已儲存到: {gif_path}")


if __name__ == "__main__":
    env = SuikaEnv(level=4)
    run_and_save_gif(env, num_steps=200, gif_path="suika_demo.gif", fps=120, seed=42)

    env.close()