from stable_baselines3 import PPO
import wandb
import datetime
from feature_extractor import MyCombinedExtractor
from wandb_callback import WandbLoggingCallback
from utils import set_seeds, make_env

config = {
    "env_name": "suika-game-l1-v0",
    "policy_type": "MultiInputPolicy",
    "total_timesteps": 3000000,
    "batch_size": 64,
    "seed": 42,
}

set_seeds(config["seed"])

id = datetime.datetime.now().strftime("%m-%d_%H-%M")
run = wandb.init(
    project="suika-sb3-ppo",
    name=id,
    config=config,
    settings=wandb.Settings(x_disable_stats=True),
)


env = make_env(config["env_name"], config["seed"])

policy_kwargs = dict(
    features_extractor_class=MyCombinedExtractor,
)

model = PPO(
    config["policy_type"],
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    batch_size=config["batch_size"],
    seed=config["seed"],
)
model.learn(
    total_timesteps=config["total_timesteps"],
    log_interval=10,  # episode
    progress_bar=True,
    callback=WandbLoggingCallback(
        eval_env=make_env(config["env_name"], config["seed"]),
        save_dir=f"weights/sb3_ppo/{id}",
        log_interval=1000,
        verbose=1,
    ),
)
run.finish()
