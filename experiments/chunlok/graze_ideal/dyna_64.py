import os
import sys
sys.path.append(os.getcwd())

from experiment_utils.sweep_configs.generate_configs import get_sorted_configuration_list_from_dict
from experiment_utils.sweep_configs.common import get_configuration_list_from_file_path
from experiment_utils.data_io.configs import check_config_completed, get_folder_name, DB_FOLDER, load_all_db_configs_and_keys, load_data_from_config_zodb
from experiment_utils.data_io.io.zodb_io import BatchDBAccess
from src.utils.run_utils import experiment_completed
import numpy as np

# get_configuration_list function is required for 
def get_configuration_list():
    parameter_dict = {
        # Determiens which folder the experiment gets saved in
        "experiment_name": ["cc_graze_ideal"],
        # Environment/Experiment
        "problem": ["GrazingWorldAdam"],
        "reward_schedule": ["cyclic"],
        "episodes": [0],
        'max_steps': [20000],
        "reward_sequence_length" : [3200],
        # Logging
        'log_keys': [('max_reward_rate', 'reward_rate')],
        'step_logging_interval': [10],
        # Seed
        "seed": list(range(15)),
        # Agent
        # 'alpha': [1.0],
        'alpha': [1.0, 0.9, 0.7, 0.6],
        "agent": ["Dyna_Tab"],
        "behaviour_alg": ["QLearner"],
        "epsilon": [0.1],
        "exploration_phase": [0],
        "gamma": [1.0],
        # "kappa": [0.05], 
        "kappa": list(np.linspace(0.01, 0.1, 10)),
        "model_planning_steps": [0],
        "no_reward_exploration": [False],
        "option_alg": ["None"],
        "planning_alg": ['Standard'],
        "planning_steps": [64],
        "search_control": ["random"],
        'learn_model': [False],
    }

    parameter_list = get_sorted_configuration_list_from_dict(parameter_dict)

    # Assuming that they all go in the same folder
    # db_folder = get_folder_name(parameter_list[0], DB_FOLDER)

    # def incomplete_filter(config):
    #     return not check_config_completed(config)

    # # Assuming all configs belong to the same folder so we aren't opening/closing the DB over and over again to check each parameter
    # with BatchDBAccess(db_folder):
    #     incomplete_parameter_list = list(filter(incomplete_filter, parameter_list))


    return parameter_list
