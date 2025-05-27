import numpy as np
import tkinter as tk
from suika_gym import SuikaEnv
import imageio
import time
import pygame
from wrappers import CoordSizeToImage

pygame.init()
window_size = (96, 96)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Live Image")
clock = pygame.time.Clock()

np.random.seed(0)
env = SuikaEnv(level=1)  # render_mode="human"
env = CoordSizeToImage(env)

boards = []
step_cnt = 0
start = time.time()
scores = []

for i in range(1):
    s, _ = env.reset()
    # frames.append(s["image"])

    done = False
    score = 0
    while not done:
        s, r, done, _, _ = env.step([np.random.rand()])
        score += r
        for board in s["boards"]:
            img = np.expand_dims(board.transpose(0, 1), -1)
            img = np.transpose(np.concat([img] * 3, -1), (1, 0, 2))
            pygame.surfarray.pixels3d(screen)[:, :, :] = img
            pygame.display.flip()
            clock.tick(30)
        step_cnt += 1

    print(score)
    scores.append(score)

end = time.time()
print(scores)
print(np.mean(scores))
print("Time taken:", end - start)
print("Steps taken:", step_cnt)
print("FPS:", step_cnt / (end - start))

# imageio.mimsave("game.gif", frames, fps=1)
