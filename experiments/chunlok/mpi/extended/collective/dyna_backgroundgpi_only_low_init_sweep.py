import os
import sys
sys.path.append(os.getcwd())

from experiment_utils.sweep_configs.generate_configs import get_sorted_configuration_list_from_dict
import numpy as np 

# get_configuration_list function is required for 
def get_configuration_list():
    parameter_dict = {
        "db_folder": ["extended_grazingworld_sweep"],
        "agent": ["Dyna_Tab"],
        "problem": ["GrazingWorldAdam"],
        "reward_schedule": ["cyclic"],
        'step_logging_interval': [1],
        "episodes": [0],
        'max_steps': [25600],
        "seed": list(range(5)),
        "exploration_phase": [0],
        "no_reward_exploration": [False],
        "alpha": [1.0, 0.9, 0.7],
        # "q_init": [-1.0],
        "epsilon": [0.1],
        "behaviour_alg": ["QLearner"],
        "search_control": ["random"],
        "option_alg": ["OnlyBackground"],
        "gamma": [1.0],
        "kappa": list(np.linspace(0.001, 0.1, 10)),
        # "kappa": [0.023000000000000003],
        "lambda": [0.9],
        "planning_steps": [4],
        "model_planning_steps": [0],
        "reward_sequence_length" : [6400],
        "planning_alg": ['Standard']
    }

    parameter_list = get_sorted_configuration_list_from_dict(parameter_dict)
    return parameter_list
