import os
import sys
sys.path.append(os.getcwd())

from experiment_utils.sweep_configs.generate_configs import get_sorted_configuration_list_from_dict

# get_configuration_list function is required for 
def get_configuration_list():
    parameter_dict = {
        # Determines which folder the experiment gets saved in
        "db_folder": ["GSP_test_no_direct"],
        'run_path': ['src/run_experiment.py'],
        
        #Environment/Experiment
        "problem": ["HMaze"],
        "episodes": [0],
        'max_steps': [8000*10],
        "reward_sequence_length" : [8000 * 1],
        'exploration_phase': [8000],
        'gamma': [0.95],

        # Logging
        'log_keys': [('max_reward_rate', 'reward_rate', 'Q')],
        'step_logging_interval': [100],

        # Seed
        "seed": [10],
        
        # Agent
        "agent": ["GSP_Tab"],
        'step_size': [1.0],
        # 'step_size': [1.0],
        'epsilon': [0.1],
        'kappa': [0.02],
        'search_control': ['current'],
        'use_pretrained_model': [False],
        'oci_steps': [1],
        'use_direct_rl': [False]
    }

    parameter_list = get_sorted_configuration_list_from_dict(parameter_dict)
    return parameter_list
