import time
import argparse
import gymnasium as gym
from suika_gym import GAME_IDS, register_envs
from stable_baselines3.common.vec_env import SubprocVecEnv

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
# from wrapper import ResizeGrayWrapper

# register_envs()

    

def make_env(game_id, seed=0, render_mode=None, render_fps=60, fps=120):
    """
    Utility to create a single environment with Monitor wrapper.
    """
    def _init():
        env = gym.make(
            game_id,
            render_mode=render_mode,
            render_fps=render_fps,
            fps=fps
        )
        env = Monitor(env)
        # env = ResizeGrayWrapper(env, width=84, height=112)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return _init



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
    parser.add_argument('--timesteps', type=int, default=5_000_000,
                        help='Total training timesteps (default=1e6).')
    parser.add_argument('--save_path', type=str, default='./models/sac_suika',
                        help='Directory to save model and logs.')
    args = parser.parse_args()

    # Create environment based on level
    game_id = GAME_IDS[args.level]
    render_mode = "human" if args.render else None

    N = 32
    env_fns = [make_env(
        game_id,
        seed=i,
        render_mode=render_mode,
        render_fps=args.render_fps,
        fps=args.fps
    ) for i in range(N)]

    vec_env = SubprocVecEnv(env_fns)

    checkpoint_callback = CheckpointCallback(save_freq=100_000, save_path=args.save_path,
                                             name_prefix='sac_checkpoint')


    eval_env = gym.make(game_id, render_mode=None, render_fps=args.render_fps, fps=args.fps)
    eval_env = Monitor(eval_env, filename="logs/monitor.csv")
    eval_callback = EvalCallback(eval_env, best_model_save_path=args.save_path,
                                 log_path=args.save_path, eval_freq=10_000,
                                 deterministic=True, render=False)

    model = SAC(
        policy="MultiInputPolicy",
        env = vec_env,
        verbose=0,
        tensorboard_log=f"{args.save_path}/tb_logs",
        device='auto',
        buffer_size=10000
    )

    current_time = time.time()

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback], progress_bar=True
    )


    model.save(f"{args.save_path}/sac_suika_final")
    print(f"Elapsed time: {(time.time() - current_time):.2f} seconds")

    vec_env.close()