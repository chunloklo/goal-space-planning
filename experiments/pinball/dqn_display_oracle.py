import os
import sys
sys.path.append(os.getcwd())

from experiment_utils.sweep_configs.generate_configs import get_sorted_configuration_list_from_dict

# get_configuration_list function is required for 
def get_configuration_list():
    parameter_dict = {
        # Determines which folder the experiment gets saved in
        "db_folder": ["pinball_impl_test"],
        'run_path': ['src/pinball_experiment.py'],
        
        #Environment/Experiment
        "problem": ["PinballOracleProblem"],
        'pinball_configuration_file': ['src/environments/data/pinball/pinball_simple_single.cfg.txt'],
        "episodes": [0],
        'max_steps': [300000],
        'explore_env': [False],
        'exploration_phase': [0],
        'gamma': [0.95],
        'render': [True],

        # Logging
        'log_keys': [('reward_rate', 'num_steps_in_ep', 'q_map')],
        'step_logging_interval': [100],

        # Seed
        "seed": [10000],
        
        # Agent
        "agent": ["Dyna_NN"],
        'step_size': [1e-3],
        'epsilon': [0.1],
        'behaviour_alg': ['DQN'],
        'polyak_stepsize': [0.1],
        'batch_num': [4],
        'batch_size':[16],

        'load_behaviour_name': ['QRC'],
    }

    parameter_list = get_sorted_configuration_list_from_dict(parameter_dict)
    return parameter_list
