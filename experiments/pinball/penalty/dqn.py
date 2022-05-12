import os
import sys
sys.path.append(os.getcwd())

from experiment_utils.sweep_configs.generate_configs import get_sorted_configuration_list_from_dict

# get_configuration_list function is required for 
def get_configuration_list():
    parameter_dict = {
        # Determines which folder the experiment gets saved in
        "db_folder": ["pinball_penalty_test"],
        'run_path': ['src/pinball_experiment.py'],
        
        #Environment/Experiment
        "problem": ["PinballSuboptimalPenaltyProblem"],
        'pinball_configuration_file': ['src/environments/data/pinball/pinball_simple_single.cfg.txt'],
        "episodes": [0],
        'max_steps': [200000],
        'explore_env': [False],
        'exploration_phase': [0],
        'gamma': [0.999],
        'render': [False],

        # Logging
        'log_keys': [('reward_rate', 'num_steps_in_ep', 'q_map')],
        'step_logging_interval': [100],

        # Seed
        "seed": [10000],
        
        # Agent
        "agent": ["Dyna_NN"],
        'epsilon': [0.1],
        'behaviour_alg': ['DQN'],
        'batch_num': [4],
        'batch_size':[16],
        'polyak_stepsize': [0.05],
        'step_size': [1e-3],
        'arch_flag': ['pinball_simple'],

    }

    parameter_list = get_sorted_configuration_list_from_dict(parameter_dict)
    return parameter_list
