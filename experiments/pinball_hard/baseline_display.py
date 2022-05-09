import os
import sys
sys.path.append(os.getcwd())

from experiment_utils.sweep_configs.generate_configs import get_sorted_configuration_list_from_dict

# This file is used for pre-learning the model.

# get_configuration_list function is required for 
def get_configuration_list():
    parameter_dict = {
        # Determines which folder the experiment gets saved in
        "db_folder": ["pinball_hard_debug"],
        'run_path': ['src/pinball_experiment.py'],
        
        #Environment/Experiment
        "problem": ["PinballHardProblem"],
        'pinball_configuration_file': ['src/environments/data/pinball/pinball_hard_single_modified.cfg.txt'],
        'explore_env': [False],
        "episodes": [0],
        'max_steps': [400000],
        'exploration_phase': [0],
        'gamma': [0.99],
        'render': [True],

        # Logging
        'log_keys': [('reward_rate', 'num_steps_in_ep')],
        'step_logging_interval': [100],

        # Seed
        "seed": [10],
        
         # Agent
        "agent": ["GSP_NN"],
        
        # Behaviour agent specific configs
        'behaviour_alg': ['DQN'],
        'polyak_stepsize': [0.1],
        'step_size': [5e-4],
        'adam_eps': [1e-8],
        'batch_num': [16],
        'batch_size': [16],
        'epsilon': [0.1],

        # Arch flags
        'behaviour_arch_flag': ['pinball_hard'],
        'model_arch_flag': ['pinball_hard'],

        # Sanity Check Steps
        'load_behaviour_as_goal_values': ['pinballhard_baseline'],
        'behaviour_goal_value_mode': ['direct'],

        # Goal Estimate Configs
        'goal_estimate_batch_size': [256],
        'goal_estimate_update_interval': [256],
        'goal_estimate_step_size': [0.005],

        # Goal space planning configs
        'use_goal_values': [True],

        # oci configs
        'use_oci_target_update': [True],
        'oci_beta': [0.0],
        # 'oci_update_interval': [16],
        # 'oci_batch_num': [4],
        # 'oci_batch_size': [32],

        # Exploration
        'use_exploration_bonus': [False],

        # Pretrain goal values:
        # 'use_pretrained_goal_values': [True],
        'use_pretrained_goal_values_optimization': [True],
        'batch_buffer_add_size': [1024],
        
        # Model training
        'goal_learner_step_size': [1e-3],
        'goal_learner_batch_num': [2],

        'learn_model_mode': ['fixed'],

        # 'learn_select_goal_models': [(2,)]
        'load_behaviour_name': ['pinballhard_baseline'],
    }

    parameter_list = get_sorted_configuration_list_from_dict(parameter_dict)
    return parameter_list
