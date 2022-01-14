import os
import sys
sys.path.append(os.getcwd())

from ParameterSweep.parameters import get_sorted_parameter_list_from_dict

# get_parameter_list function is required for 
def get_parameter_list():
    parameter_dict = {
        "experiment_name": ["grazingworld_sweep"],
        "agent": ["Dyna_Tab"],
        "problem": ["GrazingWorldAdam"],
        "reward_schedule": ["goal2_switch"],
        'step_logging_interval': [1],
        "episodes": [0],
        'max_steps': [1600],
        "seed": list(range(30)),
        "exploration_phase": [0],
        "no_reward_exploration": [False],
        "alpha": [1.0, 0.9, 0.7, 0.5, 0.3],
        "q_init": [-1.0],
        "epsilon": [0.1],
        "behaviour_alg": ["QLearner"],
        "search_control": ["random"],
        "option_alg": ["OnlyBackground"],
        "gamma": [1.0],
        "kappa": [0.3],
        "lambda": [0.9],
        "planning_steps": [4],
        "model_planning_steps": [0],
        "reward_sequence_length" : [800],
        "planning_alg": ['Standard']
    }

    parameter_list = get_sorted_parameter_list_from_dict(parameter_dict)
    return parameter_list
