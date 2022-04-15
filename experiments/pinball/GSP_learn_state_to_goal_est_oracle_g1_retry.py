import os
import sys
sys.path.append(os.getcwd())

from experiment_utils.sweep_configs.generate_configs import get_sorted_configuration_list_from_dict

# This file is used for pre-learning the model.

# get_configuration_list function is required for 
def get_configuration_list():
    parameter_dict = {
        # Determines which folder the experiment gets saved in
        "db_folder": ["pinball_impl_test"],
        'run_path': ['src/pinball_experiment.py'],
        
        #Environment/Experiment
        "problem": ["PinballOracleProblem"],
        'pinball_configuration_file': ['src/environments/data/pinball/pinball_simple_single.cfg.txt'],
        'explore_env': [True],
        "episodes": [0],
        # 'max_steps': [20000],
        'max_steps': [300000],
        'exploration_phase': [0],
        'gamma': [0.95],
        'render': [False],

        # Logging
        'log_keys': [('reward_rate', 'goal_q_map', 'goal_r_map', 'goal_gamma_map', 'reward_loss', 'policy_loss', 'step_goal_gamma_map')],
        # 'log_keys': [('reward_rate', 'goal_q_map', 'goal_r_map', 'goal_gamma_map', 'goal_r', 'goal_gamma', 'goal_baseline', 'goal_init', 'goal_values')],
        'step_logging_interval': [100],

        # Seed
        "seed": [10],
        
        # Agent
        "agent": ["GSP_NN"],
        'step_size': [1.0],
        'epsilon': [1.0],
        'kappa': [0.0],
        'search_control': ['random'],
        # 'prefill_buffer_time': [400000],
        # 'use_pretrained_behavior': [True],
        'use_pretrained_model': ['oracle_testing_2'],
        # 'use_prefill_buffer': ['oracle_prefill'],
        'OCI_update_interval': [0], ## Not needed, but still requried
        'save_state_to_goal_estimate': ['oracle_testing_retry'],
        'polyak_stepsize': [0.000],
        'learn_model_only': [True],
    }

    parameter_list = get_sorted_configuration_list_from_dict(parameter_dict)
    return parameter_list
