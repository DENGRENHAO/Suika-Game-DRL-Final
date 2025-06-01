from stable_baselines3 import SAC,DDPG,HER,DQN,TD3,PPO
from sb3_contrib import QRDQN,RecurrentPPO,TQC,TRPO
import wandb
import datetime
from feature_extractor import MyCombinedExtractor
from wandb_callback import WandbLoggingCallback
from utils import set_seeds, make_env

def model(config, env, policy_kwargs):
    print(config)
    if config["model"] == "RecurrentPPO":
        return RecurrentPPO(
            config["policy_type"],
            env,
            policy_kwargs=policy_kwargs,
            batch_size=config["batch_size"],
            seed=config["seed"],
        )
    
    elif config["model"] == "TQC":
        return TQC(
            config["policy_type"],
            env,
            policy_kwargs=policy_kwargs,
            buffer_size=50000,
            batch_size=config["batch_size"],
            seed=config["seed"],
        )
    
    elif config["model"] == "TRPO":
        return TRPO(
            config["policy_type"],
            env,
            policy_kwargs=policy_kwargs,
            batch_size=config["batch_size"],
            seed=config["seed"],
        ),

    elif config["model"] == "TD3":
        return TD3(
            config["policy_type"],
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            buffer_size=config["buffer_size"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            learning_starts=config["learning_starts"],
            seed=config["seed"],
        ),

    elif config["model"] == "SAC":
        return SAC(
            config["policy_type"],
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            buffer_size=config["buffer_size"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            learning_starts=config["learning_starts"],
            seed=config["seed"],
        ),

    elif config["model"] == "PPO":
        return PPO(
            config["policy_type"],
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            batch_size=config["batch_size"],
            seed=config["seed"],
        )

    else:
        print("undefined model, check model.py")
        return None
