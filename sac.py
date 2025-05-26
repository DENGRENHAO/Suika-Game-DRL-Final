from wrappers import CoordSizeToImage
import gymnasium as gym
from stable_baselines3 import SAC
import wandb
import datetime
from feature_extractor import MyCombinedExtractor
from wandb_callback import WandbLoggingCallback

config = {
    "env_name": "suika-game-l1-v0",
    "policy_type": "MultiInputPolicy",
    "total_timesteps": 800000,
    "buffer_size": 50000,
    "batch_size": 128,
    "learning_rate": 3e-4,
    "learning_starts": 10000,
    "log_entries": [
        "train/actor_loss",
        "train/critic_loss",
        "train/ent_coef",
    ],
}
id = datetime.datetime.now().strftime("%m-%d_%H-%M")
run = wandb.init(
    project="suika-sb3-sac",
    name=id,
    config=config,
    settings=wandb.Settings(x_disable_stats=True),
)


def make_env():
    env = gym.make(config["env_name"], render_mode="rgb_array")
    env = CoordSizeToImage(env=env)
    return env


env = make_env()

policy_kwargs = dict(
    features_extractor_class=MyCombinedExtractor,
)

model = SAC(
    config["policy_type"],
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    buffer_size=config["buffer_size"],
    batch_size=config["batch_size"],
    learning_rate=config["learning_rate"],
    learning_starts=config["learning_starts"],
)
model.learn(
    total_timesteps=config["total_timesteps"],
    log_interval=10,  # episode
    progress_bar=True,
    callback=WandbLoggingCallback(
        eval_env=make_env(),
        save_dir=f"weights/sb3_sac/{id}",
        log_interval=500,
        log_entries=config["log_entries"],
        verbose=1,
    ),
)
run.finish()
