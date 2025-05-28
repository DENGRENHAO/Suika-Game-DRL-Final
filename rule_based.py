import numpy as np
from suika_gym import SuikaEnv
import time
from wrappers import CoordSizeToImage, GRAYS
from tqdm import tqdm

## render
# import pygame
# pygame.init()
# window_size = (96, 96)
# screen = pygame.display.set_mode(window_size)
# pygame.display.set_caption("Live Image")
# clock = pygame.time.Clock()

env = SuikaEnv(level=1, seed=42, n_frames=1)  # render_mode="human"
env = CoordSizeToImage(env)

boards = []
step_cnt = 0
start = time.time()
scores = []

for i in tqdm(range(100)):
    s, _ = env.reset()
    # frames.append(s["image"])

    done = False
    score = 0
    while not done:
        board = s["boards"][-1]
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
            if board[r, c] == GRAYS[s["cur_fruit"]]:
                pos_index = dc
                # print(f"found fruit")  # at {dc} with depth {dr}")
                break
        if pos_index is None:
            pos_index = np.argmax(depths)
            # print(f"max depth")  # at {pos_index} with value {depths[pos_index]}")

        pos = pos_index / len(depths - 1)

        # print(pos)

        s, r, done, _, _ = env.step([pos])
        score += r

        ## render
        # for board in s["boards"]:
        #    img = np.expand_dims(board.transpose(0, 1), -1)
        #    img = np.transpose(np.concat([img] * 3, -1), (1, 0, 2))
        #    pygame.surfarray.pixels3d(screen)[:, :, :] = img
        #    pygame.display.flip()
        #    clock.tick(30)

        step_cnt += 1
        # time.sleep(5)

    # print(score)
    scores.append(score)

end = time.time()
# print(scores)
print("Mean Score:", np.mean(scores))
print("Std Score:", np.std(scores))
print("Time taken:", end - start)
print("Steps taken:", step_cnt)
print("FPS:", step_cnt / (end - start))

# imageio.mimsave("game.gif", frames, fps=1)
