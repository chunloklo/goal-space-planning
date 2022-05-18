import os
import sys
sys.path.append(os.getcwd())

from experiment_utils.sweep_configs.generate_configs import get_sorted_configuration_list_from_dict

# This file is used for pre-learning the model.

# get_configuration_list function is required for 
def get_configuration_list():
    parameter_dict = {
        # Determines which folder the experiment gets saved in
        "db_folder": ["pinball_refactor"],
        'run_path': ['src/pinball_experiment.py'],
        
        #Environment/Experiment
        "problem": ["PinballSuboptimalProblem"],
        'pinball_configuration_file': ['src/environments/data/pinball/pinball_simple_single.cfg.txt'],
        'explore_env': [False],
        "episodes": [0],
        'max_steps': [300000],
        'exploration_phase': [0],
        'gamma': [0.95],
        'render': [False],

        # Logging
        'log_keys': [('reward_rate', 'num_steps_in_ep')],
        'step_logging_interval': [100],

        # Seed 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29
        "seed": [0,1,2,3,4,5,6,7],
        
        # Agent
        "agent": ["DynO_FromGSP_NN"],
        
        # Behaviour agent specific configs
        'behaviour_alg': ['DQN'],
        'polyak_stepsize': [0.05],
        'step_size': [5e-3],
        'adam_eps': [1e-8],
        'batch_num': [4],
        'batch_size': [16],
        'epsilon': [0.1],
        'min_buffer_size_before_update': [10000],

        # Arch flag
        'behaviour_arch_flag': ['pinball_simple'],
        'model_arch_flag': ['pinball_simple'],

        # Goal Estimate Configs
        'goal_estimate_batch_size': [32],
        'goal_estimate_update_interval': [32],
        'goal_estimate_step_size': [0.005],

        # Goal space planning configs
        'use_goal_values': [False],
        'goal_value_init_gamma_threshold': [0.0],

        # oci configs
        'use_oci_target_update': [False],
        'use_dyno_update': [True],
        'oci_beta': [0.5],
        'oci_update_interval': [16],
        'oci_batch_num': [4],
        'oci_batch_size': [32],

        # Sanity Check Steps
        # 'load_behaviour_as_goal_values': ['q_learn'],
        # 'behaviour_goal_value_mode': ['only_values'],

        # Exploration
        'use_exploration_bonus': [False],

        # Pretrain goal values:
        # 'pretrain_goal_values': [True],
        # 'save_pretrain_goal_values': ['oracle_goal_values'],
        #'load_pretrain_goal_values': ['pinball_refactor_eps'],
        #'use_pretrained_goal_values_optimization': [True],
        'batch_buffer_add_size': [1024],
        
        # Model training
        #'save_model_name': ['dyno_pinball_easy'], # when learning model only
        'load_model_name': ['dyno_pinball_easy5'], 
        'goal_learner_step_size': [1e-3],
        'goal_learner_batch_num': [1],
        'goal_learner_batch_size': [16],
        'goal_min_buffer_size_before_update': [1000],
        'learn_model_mode': ['fixed'], # fixed: when model is loaded
        #'learn_model_mode': ['only'], # only: learn model only
        # 'learn_select_goal_models': [(15,)]
    }

    parameter_list = get_sorted_configuration_list_from_dict(parameter_dict)
    return parameter_list