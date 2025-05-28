# Suika Game - DRL Final Project

## LightZero - Sampled AlphaZero
### Pre-requisites
```
cd extern/LightZero
pip3 install -e .
```

### File Structure for Suika Game
- `extern/LightZero/zoo/suika/`
  - `config/suika_sampled_alphazero_config.py`: Configuration and the running script for Suika game.
  - `envs/suika_env.py`: Environment definition for Suika game.


### Start training
```bash
cd extern/LightZero/
python3 -u zoo/suika/config/suika_sampled_alphazero_config.py
```

### Problems
1. LightZero only provide full support for MuZero, EfficientZero, and Sampled EfficientZero. 
2. For Sampled AlphaZero, it only supports discrete action space.

In detail, in `extern/LightZero/lzero/mcts/ptree/ptree_az_sampled.py`:
1. line 246, in get_next_action: self._simulate(self.root, self.simulate_env, policy_value_func)
2. line 345, in _simulate: node.update_recursive(leaf_value, simulate_env.battle_mode_in_simulation_env)
3. line 104, in update_recursive: self.update(leaf_value)
4. line 77, in update: self._value_sum += value
  TypeError: unsupported operand type(s) for +=: 'int' and 'NoneType'
5. This is because line 321: leaf_value = self._expand_leaf_node(node, simulate_env, policy_value_func)
But in def _expand_leaf_node(...), what does this mean:
```python
if self.continuous_action_space:
    pass
```
SO, how to handle continuous action space in Sampled AlphaZero?

Error Traceback:
```
Traceback (most recent call last):
  File "/data/ddeng691/NTU_Courses/113_2/DRL/Final_Project/Suika-Game-DRL-Final/extern/LightZero/zoo/suika/config/suika_sampled_alphazero_config.py", line 110, in <module>
    train_alphazero([main_config, create_config], seed=0, max_env_step=max_env_step)
  File "/data/ddeng691/NTU_Courses/113_2/DRL/Final_Project/Suika-Game-DRL-Final/extern/LightZero/lzero/entry/train_alphazero.py", line 119, in train_alphazero
    new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/ddeng691/NTU_Courses/113_2/DRL/Final_Project/Suika-Game-DRL-Final/extern/LightZero/lzero/worker/alphazero_collector.py", line 22
1, in collect
    policy_output = self._policy.forward(obs_, temperature)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/ddeng691/anaconda3/envs/py312/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/data/ddeng691/NTU_Courses/113_2/DRL/Final_Project/Suika-Game-DRL-Final/extern/LightZero/lzero/policy/alphazero.py", line 267, in _for
ward_collect
    action, mcts_probs, root = self._collect_mcts.get_next_action(state_config_for_simulation_env_reset, self._policy_value_fn, self.collect_mcts_temperature, True)
  File "/data/ddeng691/NTU_Courses/113_2/DRL/Final_Project/Suika-Game-DRL-Final/extern/LightZero/lzero/mcts/ptree/ptree_az_sampled.py", line 246, in get_next_action
    self._simulate(self.root, self.simulate_env, policy_value_func)
  File "/data/ddeng691/NTU_Courses/113_2/DRL/Final_Project/Suika-Game-DRL-Final/extern/LightZero/lzero/mcts/ptree/ptree_az_sampled.py", line 345, in _simulate
    node.update_recursive(leaf_value, simulate_env.battle_mode_in_simulation_env)
  File "/data/ddeng691/NTU_Courses/113_2/DRL/Final_Project/Suika-Game-DRL-Final/extern/LightZero/lzero/mcts/ptree/ptree_az_sampled.py", line 104, in update_recursive
    self.update(leaf_value)
  File "/data/ddeng691/NTU_Courses/113_2/DRL/Final_Project/Suika-Game-DRL-Final/extern/LightZero/lzero/mcts/ptree/ptree_az_sampled.py", line 77, in update
    self._value_sum += value
TypeError: unsupported operand type(s) for +=: 'int' and 'NoneType'
```

## Scores

- All 0: 500-600
- random [0,1]: 681.42

## Pre-requisites

- python version: 3.12.0

```bash
pip install -r requirements.txt
```

## Environment Levels

| Level | State Representation | Game Engine Access |
| ----- | -------------------- | ------------------ |
| 1     | Coordinate-size list | Yes                |
| 2     | Coordinate-size list | No                 |
| 3     | Image                | Yes                |
| 4     | Image                | No                 |

### State Representation (Observation Space)

- Level 1, 2: Coordinate-size list

  - `Dict('grid': Box(0, 11, (57, 77, NUM_FRAMES), int8), 'next_fruit': Discrete(5))`
  - `grid`: 57x77 grid with values from 0 to 11 (0 for empty space, 1-11 for each fruit type)
  - `next_fruit`: Discrete space with 5 possible fruit types (0-4)
  - Note: grid is scaled from (570, 770) to (57, 77) by `self.grid_size = (WIDTH // 10, HEIGHT // 10)` for speed, can be changed in `suika_gym.py`.

- Level 3, 4: Image
  - `Dict('image': Box(0, 255, (770, 570, 3, NUM_FRAMES), uint8), 'next_fruit': Discrete(5))`
  - `image`: 770x570 RGB image with pixel values from 0 to 255
  - `next_fruit`: Discrete space with 5 possible fruit types (0-4)

### Game Engine Access

- Level 1, 3: Yes
  - In `info['engine_state']`, it contains a list of size `NUM_FRAMES` containing dictionaries with the following keys:
    - `id`: Unique ID of the fruit (not important, just for debugging)
    - `position`: Position of the fruit as a list of [x, y]
    - `radius`: Radius of the fruit
    - `type`: Type of the fruit (0-10)
- Level 2, 4: No
  - No access to the game engine state.

### Action Space

- `Action space: Box(0.0, 1.0, (1,), float32)`
  - Continuous action space with a single value between 0 and 1, representing the position to drop the fruit on the x-axis.

## Usage

```bash
python suika_gym.py --help
pygame 2.6.1 (SDL 2.28.4, Python 3.12.0)
Hello from the pygame community. https://www.pygame.org/contribute.html
usage: suika_gym.py [-h] [--level {1,2,3,4}] [--fps FPS] [--render] [--render_fps RENDER_FPS] [--save_gif]

Run Suika game with specified learning level

options:
  -h, --help            show this help message and exit
  --level {1,2,3,4}     Learning level (1-4) (default: 1)
  --fps FPS             Frames per second for physics simulation (default: 120)
  --render              Enable rendering
  --render_fps RENDER_FPS
                        Frames per second for rendering with pygame (default: 60)
  --save_gif            Save frames as GIF for levels 3 and 4
  --num_frames NUM_FRAMES
                        Number of intermediate frames to capture (default: 4)
```

## Auto Evaluation

- The project includes an automatic evaluation script evaluate.py located in the root directory. This script can discover and evaluate any agent that inherits from agents.base_agent.Agent.

### Usage

- Run the script from the project root directory:

  ```bash
  python evaluate.py
  ```

  Options:

  --agents AGENT_NAME: Specify one or more agent class names to evaluate (e.g., RandomAgent MyCustomAgent). If not set, all discoverable agents in the agents directory are evaluated.

  --episodes NUM_EPISODES: Number of episodes to run for each agent (default: 20).

  --max_steps MAX_STEPS_PER_EPISODE: Maximum number of steps per episode before it's considered terminated (default: 1500).

## References

- [Suika Environment by Ole-Batting](https://github.com/Ole-Batting/suika/tree/master)
