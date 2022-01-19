import os
import sys
sys.path.append(os.getcwd())

from sweep_configs.generate_configs import get_sorted_configuration_list_from_dict

# get_configuration_list function is required for 
def get_configuration_list():
    parameter_dict = {
        "experiment_name": ["extended_grazingworld_sweep"],
        "agent": ["DynaOptions_Tab"],
        "problem": ["GrazingWorldAdam"],
        "reward_schedule": ["cyclic"],
        'step_logging_interval': [1],
        "episodes": [0],
        'max_steps': [1600],
        "seed": list(range(30)),
        "exploration_phase": [0],
        "no_reward_exploration": [False],
        "alpha": [0.9, 0.7, 0.5, 0.3, 0.1],
        "epsilon": [0.1],
        "behaviour_alg": ["QLearner"],
        "option_model_alg": ['sutton'],
        "search_control": ["random"],
        "option_alg": ["None"],
        "gamma": [1.0],
        "kappa": [0.3],
        "lambda": [0.9],
        "planning_steps": [4],
        "model_planning_steps": [0],
        "reward_sequence_length" : [800],
        "planning_alg": ['Standard']
    }

    parameter_list = get_sorted_configuration_list_from_dict(parameter_dict)
    return parameter_list
