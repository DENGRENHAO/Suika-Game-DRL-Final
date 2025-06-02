import os
import numpy as np
import imageio
import gymnasium as gym
from suika_gym import SuikaEnv
from wrappers import GRAYS, CoordSizeToImage

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


    for step in range(num_steps):
        board = obs["boards"][-1]
        left_offset = env.fruit_horizontal_offset + 2
        first_nonzeros = np.argmax(board.T != 0, axis=-1)
        depths = first_nonzeros - env.wall_height_offset
        depths = depths[
            left_offset : (env.image_size[1] - env.horizontal_wall_thickness - 1)
        ]
        # print("depths:", depths)
        # print("======")
        # print(f"{s["cur_fruit"]}", end=" ")
        pos_index = None
        for dr, dc in zip(depths, range(len(depths))):
            r = dr + env.wall_height_offset
            c = dc + left_offset
            if board[r, c] == GRAYS[obs["cur_fruit"]]:
                pos_index = dc
                # print(f"found fruit")  # at {dc} with depth {dr}")
                break

        if pos_index is None:
            pos_index = np.argmax(depths)
            # print(f"max depth")  # at {pos_index} with value {depths[pos_index]}")

        pos = pos_index / len(depths - 1)

        # print(pos)

        obs, r, terminated, _, info = env.step([pos])

        # print(info)

        # print(len(info['fruit_states']))

        for i in range(len(info['fruit_states'])):

            img = env.render_states(info['fruit_states'])[i]

            frames.append(img)

        # if np.shape(img)!=(770,570,3):
        #     print(np.shape(img))

        if terminated:
            print(f"第 {step+1} 步時遊戲結束 (terminated)。")
            break

    os.makedirs(os.path.dirname(gif_path) or ".", exist_ok=True)


    imageio.mimsave(gif_path, frames, fps=fps, loop = 0)
    print(f"GIF 已儲存到: {gif_path}")


if __name__ == "__main__":
    env = SuikaEnv(level=1)
    env = CoordSizeToImage(env)
    run_and_save_gif(env, num_steps=500, gif_path="suika_demo_rb.gif", fps=20, seed=42)

    env.close()


        