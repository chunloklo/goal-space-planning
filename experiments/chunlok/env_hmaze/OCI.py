import os
import sys
sys.path.append(os.getcwd())

from experiment_utils.sweep_configs.generate_configs import get_sorted_configuration_list_from_dict

# get_configuration_list function is required for 
def get_configuration_list():
    parameter_dict = {
        # Determines which folder the experiment gets saved in
        "db_folder": ["GSP_test"],
        'run_path': ['src/run_experiment.py'],
        
        #Environment/Experiment
        "problem": ["HMaze"],
        "episodes": [0],
        'max_steps': [32000*10],
        "reward_sequence_length" : [32000 * 1],
        'exploration_phase': [0],
        'gamma': [0.95],
        'reward_schedule': [None], # Not needed in HMaze, but regardless, its here bec Dyna_Tab needs it.

        # Logging
        'log_keys': [('Q', 'reward_rate')],
        'step_logging_interval': [100],

        # Seed
        "seed": [10],
        
        # Agent
        "agent": ["Dyna_Tab"],
        "behaviour_alg": ["QLearner"],
        'alpha': [1.0],
        "epsilon": [0.1],
        "exploration_phase": [0],
        "gamma": [1.0],
        "kappa": [0.001],
        "model_planning_steps": [0],
        "no_reward_exploration": [False],
        "option_alg": ["Background"],
        "planning_alg": ['Standard'],
        "planning_steps": [1],
        "search_control": ["current"],
        'learn_model': [True],
    }

    parameter_list = get_sorted_configuration_list_from_dict(parameter_dict)
    return parameter_list
