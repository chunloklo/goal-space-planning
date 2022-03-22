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
        "problem": ["PinballProblem"],
        'pinball_configuration_file': ['src/environments/data/pinball/pinball_simple_single.cfg.txt'],
        "episodes": [0],
        'max_steps': [600000],
        'explore_env': [False],
        'exploration_phase': [0],
        'gamma': [0.99],
        'render': [False],

        # Logging
        'log_keys': [('reward_rate', 'num_steps_in_ep', 'q_map')],
        # 'log_keys': [('reward_rate', 'q_map')],
        'step_logging_interval': [100],

        # Seed
        "seed": [10],
        
        # Agent
        "agent": ["Dyna_NN"],
        'step_size': [1.0],
        'epsilon': [0.1],
        'search_control': ['random'],
        'use_pretrained_behavior': [False],
    }

    parameter_list = get_sorted_configuration_list_from_dict(parameter_dict)
    return parameter_list
