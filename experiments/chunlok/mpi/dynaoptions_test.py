import os
import sys
sys.path.append(os.getcwd())

from ParameterSweep.parameters import get_sorted_parameter_list_from_dict

# get_parameter_list function is required for 
def get_parameter_list():
    parameter_dict = {
        "experiment_name": ["representation_refactor_test"],
        "agent": ["DynaOptions_Tab"],
        "problem": ["GrazingWorldAdam"],
        "episodes": [1000],
        "seed": [0],
        "exploration_phase": [0],
        "no_reward_exploration": [False],
        "alpha": [0.9, 0.7, 0.5, 0.3, 0.1],
        "epsilon": [0.1],
        "behaviour_alg": ["QLearner"],
        "search_control": ["random"],
        "option_model_alg": ["sutton"],
        "gamma": [1.0],
        "kappa": [0.3, 0.5, 0.7, 0.9],
        "lambda": [0.9],
        "planning_steps": [5],
        "model_planning_steps": [0],
        "reward_sequence_length" : [500],
        "planning_alg": ['Standard']
    }

    parameter_list = get_sorted_parameter_list_from_dict(parameter_dict)
    return parameter_list
