import os
import sys
sys.path.append(os.getcwd())

from experiment_utils.sweep_configs.generate_configs import get_sorted_configuration_list_from_dict

# get_configuration_list function is required for 
def get_configuration_list():
    parameter_dict = {
        "db_folder": ["grazingworld_sweep"],
        "agent": ["Dyna_Tab"],
        "problem": ["GrazingWorldAdam"],
        'step_logging_interval': [1],
        "episodes": [0],
        'max_steps': [400000],
        "seed": [0],
        "exploration_phase": [400000],
        "no_reward_exploration": [False],
        "alpha": [0.9],
        "epsilon": [0.0],
        "behaviour_alg": ["QLearner"],
        "search_control": ["random"],
        "option_alg": ["None"],
        "gamma": [1.0],
        "kappa": [0.3],
        "lambda": [0.9],
        "planning_steps": [5],
        "model_planning_steps": [0],
        "reward_sequence_length" : [400000],
        "planning_alg": ['Standard']
    }

    parameter_list = get_sorted_configuration_list_from_dict(parameter_dict)
    return parameter_list
