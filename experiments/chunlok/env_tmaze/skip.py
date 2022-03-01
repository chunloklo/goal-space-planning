import os
import sys
sys.path.append(os.getcwd())

from experiment_utils.sweep_configs.generate_configs import get_sorted_configuration_list_from_dict
import numpy as np 

# get_configuration_list function is required for 
def get_configuration_list():
    parameter_dict = {
        # Determines which folder the experiment gets saved in
        "db_folder": ["tmaze_test"],
        'run_path': ['src/run_experiment.py'],

        # Environment/Experiment
        "problem": ["TMaze"],
        "episodes": [0],
        'max_steps': [19200 * 10],
        "reward_sequence_length" : [19200],

        # Logging
        'log_keys': [('max_reward_rate', 'reward_rate', 'skip_probability_weights', 'avg_greedy_policy')],
        # 'log_keys': [('max_reward_rate', 'reward_rate')],
        'step_logging_interval': [100],

        # Seed
        "seed": list(range(5)),

        # Agent
        'alpha': [1.0],
        "agent": ["Direct_Tab"],
        "behaviour_alg": ["QLearner"],
        "epsilon": [0.1],
        "exploration_phase": [0],
        "gamma": [0.9],
        "skip_action": [True],
        'initial_skip_weight': [-4],
        'skip_lr': [1e-2],
        'avg_greedy_policy_lr': [1e-3],
        'filter_class': ['common_greedy'],
    }

    parameter_list = get_sorted_configuration_list_from_dict(parameter_dict)
    return parameter_list
