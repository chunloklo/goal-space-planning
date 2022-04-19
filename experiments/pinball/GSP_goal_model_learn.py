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
        'max_steps': [100000],
        'exploration_phase': [0],
        'gamma': [0.95],
        'render': [False],

        # Logging
        'log_keys': [('reward_rate', 'goal_q_map', 'goal_r_map', 'goal_gamma_map', 'reward_loss', 'policy_loss', 'step_goal_gamma_map')],
        'step_logging_interval': [100],

        # Seed
        "seed": [10],
        
        # Agent
        "agent": ["GSP_NN"],
        'epsilon': [1.0],
        'kappa': [0.0],
        'search_control': ['random'],
        'polyak_stepsize': [0.001],
        'step_size': [1e-4],
        'adam_eps': [1e-8],
        'batch_size': [16],
        'use_pretrained_model': [None],
        'use_prefill_buffer': ['GSP_prefill_400k'],
        'OCI_update_interval': [0], ## Not needed, but still requried
        'save_state_to_goal_estimate_name': ['GSP_model_800k'], # 400k is actually 800k steps. It's also been renamed to 'standard_800k' since it doesn't use the oracle goals
        'learn_model_only': [True],

        # 'learn_select_goal_models': [(15,)]
    }

    parameter_list = get_sorted_configuration_list_from_dict(parameter_dict)
    return parameter_list
