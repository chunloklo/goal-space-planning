# {
#     "experiment_name": "tmaze_test",
#     "agent": "Direct_Tab",
#     "problem": "TMaze",
#     "episodes": -1,
#     "max_steps": 200000,
#     "metaParameters": {
#         "seed": [0],
#         "exploration_phase": [0],
#         "no_reward_exploration": [true],
#         "alpha": [0.7],
#         "slow_alpha": [0.007],
#         "epsilon": [0.2],
#         "behaviour_alg": ["QLearner"],
#         "skip_alg": ["None"],
#         "gamma": [1.0],
#         "lambda": [0.9],
#         "reward_sequence_length" : [40000]
#     }
# }


import os
import sys
sys.path.append(os.getcwd())

from experiment_utils.sweep_configs.generate_configs import get_sorted_configuration_list_from_dict
import numpy as np 

# get_configuration_list function is required for 
def get_configuration_list():
    parameter_dict = {
        # Determiens which folder the experiment gets saved in
        "experiment_name": ["tmaze_test"],
        # Environment/Experiment
        "problem": ["TMaze"],
        "episodes": [0],
        'max_steps': [120000],
        "reward_sequence_length" : [20000],
        # Logging
        'log_keys': [('Q', 'max_reward_rate', 'reward_rate', 'one_step_suboptimality', 'avg_suboptimality')],
        'step_logging_interval': [10],
        # Seed
        "seed":  [0],
        # Agent
        'alpha': [1.0],
        "agent": ["Direct_Tab"],
        "behaviour_alg": ["QLearner"],
        "epsilon": [0.1],
        "exploration_phase": [0],
        "gamma": [0.9],
    }

    parameter_list = get_sorted_configuration_list_from_dict(parameter_dict)
    return parameter_list
