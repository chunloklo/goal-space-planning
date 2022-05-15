import os
import sys
sys.path.append(os.getcwd())

from experiment_utils.sweep_configs.generate_configs import get_sorted_configuration_list_from_dict

# This file is used for pre-learning the model.

# get_configuration_list function is required for 
def get_configuration_list():
    parameter_dict = {
        # Determines which folder the experiment gets saved in
        "db_folder": ["pinball_refactor_test"],
        'run_path': ['src/pinball_experiment.py'],
        
        #Environment/Experiment
        "problem": ["PinballHardProblem"],
        'pinball_configuration_file': ['src/environments/data/pinball/pinball_hard_single_modified.cfg.txt'],
        'explore_env': [False],
        "episodes": [0],
        'max_steps': [300000],
        'exploration_phase': [0],
        'gamma': [0.99],
        'render': [True],

        # Logging
        'log_keys': [('reward_rate', 'num_steps_in_ep')],
        'step_logging_interval': [100],

        # Seed
        "seed": list(range(8)),
        
         # Agent
        "agent": ["GSP_NN"],
        
        # Behaviour agent specific configs
        'behaviour_alg': ['DQN'],
        'polyak_stepsize': [0.025],
        'step_size': [5e-4],
        'adam_eps': [1e-8],
        'batch_num': [4],
        'batch_size': [16],
        'epsilon': [0.1],
        'min_buffer_size_before_update': [1000],

        # Sanity Check Steps
        # 'load_behaviour_as_goal_values': ['pinballhard_baseline'],
        # 'behaviour_goal_value_mode': ['only_values'],

        # Arch flag
        'behaviour_arch_flag': ['pinball_hard'],
        'model_arch_flag': ['pinball_hard'],

        # Goal Estimate Configs
        'goal_estimate_batch_size': [256],
        'goal_estimate_update_interval': [256],
        'goal_estimate_step_size': [0.005],

        # Goal space planning configs
        'use_goal_values': [True],
        'goal_value_init_gamma_threshold': [0.0],

        # oci configs
        'use_oci_target_update': [True],
        'oci_beta': [1.0, 0.5, 0.3, 0.2, 0.1, 1e-3, 1e-5, 1e-7],

        # Exploration
        'use_exploration_bonus': [False],

        # Pretrain goal values:
        # 'use_pretrained_goal_values': [True],
        'load_pretrain_goal_values': ['pinball_hard_ddqn_value_policy_8_goals'],
        'use_pretrained_goal_values_optimization': [True],
        'batch_buffer_add_size': [1024],
        
        # Model training
        'load_model_name': ['pinball_hard_ddqn_value_policy_8_goals'],
        'goal_learner_step_size': [5e-4],
        'goal_learner_batch_num': [1],
        'goal_learner_batch_size': [16],
        'goal_min_buffer_size_before_update': [10000],
        'learn_model_mode': ['fixed'],
    }

    parameter_list = get_sorted_configuration_list_from_dict(parameter_dict)
    return parameter_list
