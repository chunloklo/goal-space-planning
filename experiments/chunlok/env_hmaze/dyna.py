import os
import sys
sys.path.append(os.getcwd())

from experiment_utils.sweep_configs.generate_configs import get_sorted_configuration_list_from_dict

# get_configuration_list function is required for 
def get_configuration_list():
    parameter_dict = {
        # Determines which folder the experiment gets saved in
        "db_folder": ["2022_03_07_small_sweep"],
        'run_path': ['src/run_experiment.py'],
        
        #Environment/Experiment
        "problem": ["HMaze"],
        "episodes": [0],
        'max_steps': [8000*10],
        "reward_sequence_length" : [8000 * 1],
        'exploration_phase': [8000],
        'gamma': [0.95],
        'reward_schedule': [None], # Not needed in HMaze, but regardless, its here bec Dyna_Tab needs it.

        # Logging
        'log_keys': [('max_reward_rate', 'reward_rate', 'Q')],
        'step_logging_interval': [100],

        # Seed
        "seed": [10],
        
        # Agent
        "agent": ["Dyna_Tab"],
        'alpha': [1.0],
        "behaviour_alg": ["QLearner"],
        "epsilon": [0.1],
        "gamma": [1.0],
        # "kappa": [0.03],
        'kappa': [0.001],
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
