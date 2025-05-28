from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 5
num_simulations = 25
update_per_collect = 50
batch_size = 256
max_env_step = int(2e5)
mcts_ctree = False
continuous_action_space = True
K = 5  # num_of_sampled_actions
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================
suika_alphazero_config = dict(
    exp_name=f'data_az_ctree/suika_alphazero_play-with-bot-mode_ns{num_simulations}_upc{update_per_collect}_seed0',
    env=dict(
        board_size=96,
        battle_mode='play_with_bot_mode',
        bot_action_type='v0',  # {'v0', 'alpha_beta_pruning'}
        channel_last=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        # ==============================================================
        # for the creation of simulation env
        # ==============================================================
    ),
    policy=dict(
        mcts_ctree=mcts_ctree,
        # ==============================================================
        # for the creation of simulation env
        simulation_env_id='suika',
        simulation_env_config_type='play_with_bot_mode',
        # ==============================================================
        torch_compile=False,
        tensor_float_32=False,
        model=dict(
            observation_shape=(4, 96, 96),
            action_space_size=int(1),
            # We use the small size model for suika.
            num_res_blocks=1,
            num_channels=16,
            value_head_hidden_channels=[8],
            policy_head_hidden_channels=[8],
            continuous_action_space=continuous_action_space,
            num_of_sampled_actions=K,
            discrete_action_encoding_type='one_hot',
            norm_type='BN', 
        ),
        sampled_algo=True,
        cuda=False,
        board_size=96,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        grad_clip_value=0.5,
        value_weight=1.0,
        entropy_weight=0.0,
        n_episode=n_episode,
        eval_freq=int(2e3),
        mcts=dict(
            num_simulations=num_simulations,
            continuous_action_space=continuous_action_space,
            legal_actions=None,
            # (int) The action space size.
            action_space_size=int(1),
            # (int) The number of sampled actions for each state.
            num_of_sampled_actions=K,
        ),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

suika_alphazero_config = EasyDict(suika_alphazero_config)
main_config = suika_alphazero_config

suika_alphazero_create_config = dict(
    env=dict(
        type='suika',
        import_names=['zoo.suika.envs.suika_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='alphazero',
        import_names=['lzero.policy.alphazero'],
    ),
    collector=dict(
        type='episode_alphazero',
        import_names=['lzero.worker.alphazero_collector'],
    ),
    evaluator=dict(
        type='alphazero',
        import_names=['lzero.worker.alphazero_evaluator'],
    )
)
suika_alphazero_create_config = EasyDict(suika_alphazero_create_config)
create_config = suika_alphazero_create_config

if __name__ == '__main__':
    from lzero.entry import train_alphazero
    train_alphazero([main_config, create_config], seed=0, max_env_step=max_env_step)
