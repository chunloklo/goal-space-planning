import os
import sys
sys.path.append(os.getcwd())

from experiment_utils.sweep_configs.generate_configs import get_sorted_configuration_list_from_dict

# This file is used for pre-learning the model.

# get_configuration_list function is required for 
def get_configuration_list():
    parameter_dict = {
        # Determines which folder the experiment gets saved in
        "db_folder": ["pinball_penalty_test"],
        'run_path': ['src/pinball_experiment.py'],
        
        #Environment/Experiment
        "problem": ["PinballSuboptimalPenaltyProblem"],
        'pinball_configuration_file': ['src/environments/data/pinball/pinball_simple_single.cfg.txt'],
        'explore_env': [True],
        "episodes": [0],
        'max_steps': [300000],
        'exploration_phase': [0],
        'gamma': [0.95],
        'render': [False],

        # Logging
        'log_keys': [('reward_rate',)],
        'step_logging_interval': [100],

        # Seed
        "seed": [10000],
        
         # Agent
        "agent": ["GSP_NN"],
        
        # Behaviour agent specific configs
        'behaviour_alg': ['DQN'],
        'polyak_stepsize': [0.1],
        'step_size': [1e-3],
        'adam_eps': [1e-8],
        'batch_num': [4],
        'batch_size': [16],
        'epsilon': [1.0],
        'min_buffer_size_before_update': [10000],

        # Arch flags
        'behaviour_arch_flag': ['pinball_simple_experimental'],
        'model_arch_flag': ['pinball_simple_experimental'],

        # Sanity Check Steps
        # 'load_behaviour_as_goal_values': ['q_learn'],
        # 'behaviour_goal_value_mode': ['direct'],

        # Goal Estimate Configs
        'goal_estimate_batch_size': [256],
        'goal_estimate_update_interval': [256],
        'goal_estimate_step_size': [0.005],

        # Goal space planning configs
        'use_goal_values': [True],
        'goal_value_init_gamma_threshold': [0.6],

        # oci configs
        'use_oci_target_update': [True],
        'oci_beta': [0.0],
        # 'oci_update_interval': [16],
        # 'oci_batch_num': [4],
        # 'oci_batch_size': [32],

        # Exploration
        'use_exploration_bonus': [False],

        # Pretrain goal values:
        # 'use_pretrained_goal_values': [True],
        'use_pretrained_goal_values_optimization': [True],
        'batch_buffer_add_size': [1024],
        
        # Model training
        'save_model_name': ['pinball_penalty_debug_exp'], # throwaway
        'goal_learner_step_size': [1e-3],
        'goal_learner_batch_num': [1],
        'goal_learner_batch_size': [16],
        'goal_min_buffer_size_before_update': [10000],
        'learn_model_mode': ['only'],
        'goal_learner_alg': ['DDQN'],
        'use_reward_for_model_policy': [True],
        'leave_init_value': [-20.0],

        'learn_select_goal_models': [(2, )]
    }
    parameter_list = get_sorted_configuration_list_from_dict(parameter_dict)
    return parameter_list
