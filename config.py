config = {
    "RecurrentPPO":{
        "env_name": "suika-game-l1-v0",
        "policy_type": "MultiInputLstmPolicy",
        "total_timesteps": 3000000,
        "batch_size": 128,
        "seed": 42,
        "model": "RecurrentPPO",
        "env_num" : 32
    },
    "TQC":{
        "env_name": "suika-game-l1-v0",
        "policy_type": "MultiInputPolicy",
        "total_timesteps": 3000000,
        "batch_size": 128,
        "seed": 42,
        "model": "TQC",
        "env_num" : 32
    },
    
    "TRPO":{
        "env_name": "suika-game-l1-v0",
        "policy_type": "MultiInputPolicy",
        "total_timesteps": 3000000,
        "batch_size": 128,
        "seed": 42,
        "model": "TRPO",
        "env_num" : 16
    },
    "TD3":{
        "env_name": "suika-game-l1-v0",
        "policy_type": "MultiInputPolicy",
        "total_timesteps": 800000,
        "buffer_size": 50000,
        "batch_size": 128,
        "learning_rate": 5e-4,
        "learning_starts": 10000,
        "log_entries": [
            "train/actor_loss",
            "train/critic_loss",
        ],
        "seed": 42,
    },
    "SAC":{
        "env_name": "suika-game-l1-v0",
        "policy_type": "MultiInputPolicy",
        "total_timesteps": 600000,
        "buffer_size": 50000,
        "batch_size": 128,
        "learning_rate": 3e-4,
        "learning_starts": 10000,
        "log_entries": [
            "train/actor_loss",
            "train/critic_loss",
            "train/ent_coef",
        ],
        "seed": 42,
    },
    "PPO":{
        "env_name": "suika-game-l1-v0",
        "policy_type": "MultiInputPolicy",
        "total_timesteps": 2000000,
        "batch_size": 64,
        "seed": 42,
    },
}