import os
import sys
sys.path.append(os.getcwd())

from experiment_utils.sweep_configs.generate_configs import get_sorted_configuration_list_from_dict
import numpy as np 

# get_configuration_list function is required for 
def get_configuration_list():
    parameter_dict = {
        # Determiens which folder the experiment gets saved in
        "experiment_name": ["graze_ideal"],
        # Environment/Experiment
        "problem": ["GrazingWorldAdam"],
        "reward_schedule": ["cyclic"],
        "episodes": [0],
        'max_steps': [20000],
        "reward_sequence_length" : [3200],
        # Logging
        'log_keys': [('max_reward_rate', 'reward_rate')],
        'step_logging_interval': [10],
        # Seed
        "seed": list(range(15)),
        # 'seed': [1],
        # Agent
        'alpha': [1.0, 0.9, 0.7, 0.6],
        # "alpha": [1.0],
        "agent": ["Dyna_Tab"],
        "behaviour_alg": ["QLearner"],
        "epsilon": [0.1],
        "exploration_phase": [0],
        "gamma": [1.0],
        # "kappa": [0.06],
        "kappa": list(np.linspace(0.01, 0.1, 10)),
        "model_planning_steps": [0],
        "no_reward_exploration": [False],
        "option_alg": ["DecisionTime"],
        "planning_alg": ['Standard'],
        "planning_steps": [0],
        "search_control": ["current"],
        'learn_model': [False],
    }

    parameter_list = get_sorted_configuration_list_from_dict(parameter_dict)
    return parameter_list
