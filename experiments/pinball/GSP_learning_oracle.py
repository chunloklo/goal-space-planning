import os
import sys
sys.path.append(os.getcwd())

from experiment_utils.sweep_configs.generate_configs import get_sorted_configuration_list_from_dict

# get_configuration_list function is required for 
def get_configuration_list():
    parameter_dict = {
        # Determines which folder the experiment gets saved in
        "db_folder": ["pinball_impl_test"],
        'run_path': ['src/pinball_experiment.py'],
        
        #Environment/Experiment
        "problem": ["PinballOracleProblem"],
        'pinball_configuration_file': ['src/environments/data/pinball/pinball_simple_single.cfg.txt'],
        'explore_env': [False],
        "episodes": [0],
        # 'max_steps': [100],
        'max_steps': [200000],
        'exploration_phase': [0],
        'gamma': [0.95],
        'render': [False],

        # Logging
        # 'log_keys': [('reward_rate', 'goal_q_map', 'goal_r_map', 'goal_gamma_map', 'reward_loss', 'policy_loss')],
        'log_keys': [('reward_rate', 'q_map', 'num_steps_in_ep', 'goal_q_map', 'goal_r_map', 'goal_gamma_map', 'goal_r', 'goal_gamma', 'goal_baseline', 'goal_init', 'goal_values')],
        # 'log_keys': [('reward_rate', 'goal_baseline', 'goal_values')],
        # 'log_keys': [('reward_rate', 'q_map', 'goal_baseline', 'goal_values')],
        'step_logging_interval': [100],

        # Seed
        "seed": [102309],
        
        # Agent
        "agent": ["GSP_NN"],
        'step_size': [1.0],
        'epsilon': [0.1],
        'search_control': ['random'],
        'OCI_update_interval': [2],
        'polyak_stepsize': [0.05],
        'use_exploration_bonus': [True],

        'use_goal_values': [True],

        # Saving/Loading of Models/Behavior:
        'save_behavior': [True],
        'use_pretrained_behavior': [False],
        'use_pretrained_model': ['oracle3'],
    }

    parameter_list = get_sorted_configuration_list_from_dict(parameter_dict)
    return parameter_list
