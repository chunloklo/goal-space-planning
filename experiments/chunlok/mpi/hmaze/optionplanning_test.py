import os
import sys
sys.path.append(os.getcwd())

from experiment_utils.sweep_configs.generate_configs import get_sorted_configuration_list_from_dict

# get_configuration_list function is required for 
def get_configuration_list():
    parameter_dict = {
        "db_folder": ["option_planning"],
        "agent": ["OptionPlanning_Tab"],
        "problem": ["HMaze"],
        "episodes": [0],
        'max_steps': [32000 * 2],
        "reward_sequence_length" : [8000],
        'step_logging_interval': [100],
        "seed": [10],
        "exploration_phase": [0],
        "no_reward_exploration": [False],
        "alpha": [1.0],
        "epsilon": [0.1],
        "search_control": ["random"],
        'planning_steps': [8],
        'goal_planning_steps': [4],
        "gamma": [0.95],
        "kappa": [0.01],

    }

    parameter_list = get_sorted_configuration_list_from_dict(parameter_dict)
    return parameter_list
