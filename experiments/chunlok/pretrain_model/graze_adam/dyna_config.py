import os
import sys
sys.path.append(os.getcwd())

from experiment_utils.sweep_configs.generate_configs import get_sorted_configuration_list_from_dict
import numpy as np 

# get_configuration_list function is required for 
def get_configuration_list():
    parameter_dict = {
        # Determiens which folder the experiment gets saved in
        "experiment_name": ["pretrain_model"],
        # Environment/Experiment
        "problem": ["GrazingWorldAdam"],
        "reward_schedule": ["zero_debug"],
        "episodes": [0],
        'max_steps': [12800],
        "reward_sequence_length" : [0],
        # Logging
        'log_keys': [frozenset({    'model_r', 
                                    'model_discount', 
                                    'model_transition',
                                    'action_model_r', 
                                    'action_model_discount', 
                                    'action_model_transition'
                    })],
        'step_logging_interval': [1000],
        # Seed
        "seed": [0],
        # Agent
        "alpha": [1.0],
        "agent": ["Dyna_Tab"],
        "behaviour_alg": ["QLearner"],
        "epsilon": [0.1],
        "exploration_phase": [128000],
        "gamma": [1.0],
        "kappa": [0.0],
        "model_planning_steps": [0],
        "no_reward_exploration": [False],
        "option_alg": ["None"],
        "planning_alg": ['Standard'],
        "planning_steps": [4],
        "search_control": ["random"],
        'learn_model': [True],
    }

    parameter_list = get_sorted_configuration_list_from_dict(parameter_dict)
    return parameter_list
