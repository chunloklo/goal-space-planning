import os
import sys
sys.path.append(os.getcwd())

from experiment_utils.sweep_configs.generate_configs import get_sorted_configuration_list_from_dict
import numpy as np 

# get_configuration_list function is required for 
def get_configuration_list():
    parameter_dict = {
        # Determiens which folder the experiment gets saved in
        "experiment_name": ["tmaze_test_sweep"],
        # Environment/Experiment
        "problem": ["TMaze"],
        "episodes": [0],
        'max_steps': [9600 * 40],
        "reward_sequence_length" : [9600 * 2],
        # Logging
        'log_keys': [('max_reward_rate', 'reward_rate')],
        # 'log_keys': [('max_reward_rate', 'reward_rate')],
        'step_logging_interval': [100],
        # Seed
        # "seed":  [10],
        "seed": list(range(5)),
        # Agent
        'alpha': [1.0, 0.9, 0.7],
        "agent": ["Direct_Tab"],
        "behaviour_alg": ["QLearner"],
        "epsilon": [0.1],
        "exploration_phase": [0],
        "gamma": [0.99],
        "skip_action": [False],
        "base_q": [True],
    }

    parameter_list = get_sorted_configuration_list_from_dict(parameter_dict)
    return parameter_list
