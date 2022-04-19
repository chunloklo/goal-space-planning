import os
import sys
sys.path.append(os.getcwd())

from experiment_utils.sweep_configs.generate_configs import get_sorted_configuration_list_from_dict

# This file is used for pre-learning the model.

# get_configuration_list function is required for 
def get_configuration_list():
    parameter_dict = {
        # Determines which folder the experiment gets saved in
        "db_folder": ["pinball_impl_test"],
        'run_path': ['src/pinball_experiment.py'],
        
        #Environment/Experiment
        "problem": ["PinballProblem"],
        'pinball_configuration_file': ['src/environments/data/pinball/pinball_simple_single.cfg.txt'],
        'explore_env': [True],
        "episodes": [0],
        'max_steps': [400000],
        'exploration_phase': [0],
        'gamma': [0.95],
        'render': [False],

        # Logging
        'log_keys': [()],
        'step_logging_interval': [100],

        # Seed
        "seed": [10],
        
        # Agent
        "agent": ["GSP_NN"],
        'step_size': [1.0],
        'epsilon': [1.0],
        'kappa': [0.0],
        'search_control': ['random'],
        'prefill_buffer_time': [400000],
        # 'use_pretrained_behavior': [True],
        # 'use_pretrained_model': ['oracle6'],
        # 'use_prefill_buffer': ['oracle_prefill'],
        # Not needed, but still required
        'batch_size': [16],
        'step_size': [1e-3],

        'OCI_update_interval': [0],
        'save_state_to_goal_estimate_name': ['GSP_prefill_400k'],
        'polyak_stepsize': [0.000],
        'learn_model_only': [True],
    }

    parameter_list = get_sorted_configuration_list_from_dict(parameter_dict)
    return parameter_list
