import os
import sys
sys.path.append(os.getcwd())

from experiment_utils.sweep_configs.generate_configs import get_sorted_configuration_list_from_dict
from experiment_utils.data_io.configs import get_incomplete_configuration_list
import numpy as np 

# get_configuration_list function is required for 
def get_configuration_list():
    parameter_dict = {
        # Determiens which folder the experiment gets saved in
        "experiment_name": ["tmaze_test_sweep"],
        'run_path': ['src/run_experiment.py'],

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
        "seed": list(range(1)),
        # Agent
        'alpha': [1.0],
        "agent": ["Direct_Tab"],
        "behaviour_alg": ["QLearner"],
        "epsilon": [0.1],
        "exploration_phase": [0],
        "gamma": [0.9],
        "skip_action": [True],
        'initial_skip_weight': [-4, -2, 0, 2, 4],
        'skip_lr': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0],
        'avg_greedy_policy_lr': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0],
    }

    parameter_list = get_sorted_configuration_list_from_dict(parameter_dict)

    return parameter_list

    incomplete_configs = get_incomplete_configuration_list(parameter_list)
    return incomplete_configs
