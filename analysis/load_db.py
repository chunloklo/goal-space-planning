
import os
import sys
sys.path.append(os.getcwd())
from experiment_utils.data_io.configs import get_incomplete_configuration_list, load_all_db_configs_and_keys, save_data_zodb
import numpy as np
import code
from experiment_utils.sweep_configs.common import get_configuration_list_from_file_path

# Mainly a testing script for loading stuff from dbs

if __name__ == "__main__":

    # create_individual_plot('experiments/chunlok/env_tmaze_sweep/skip.py')

    db_dir = 'results_dbs/cc_graze_ideal_test'

    config_file = '/Users/chunloklo/schoolwork/dyna-options/experiments/chunlok/graze_ideal/dyna_64.py'

    config_list = get_configuration_list_from_file_path(config_file)

    incomplete = get_incomplete_configuration_list(config_list)
    print(len(incomplete))

    # configs = load_all_db_configs_and_keys(db_dir)
    
    # for config in configs:
    #     print(config)

    # test_config = {'agent': 'DEBUG_TEST', 'alpha': 0.9, 'behaviour_alg': 'QLearner', 'episodes': 0, 'epsilon': 0.1, 'experiment_name': 'cc_graze_ideal', 'exploration_phase': 0, 'gamma': 1.0, 'kappa': 0.020000000000000004, 'learn_model': False, 'log_keys': ('max_reward_rate', 'reward_rate'), 'max_steps': 20000, 'model_planning_steps': 0, 'no_reward_exploration': False, 'option_alg': 'None', 'planning_alg': 'Standard', 'planning_steps': 64, 'problem': 'GrazingWorldAdam', 'reward_schedule': 'cyclic', 'reward_sequence_length': 3200, 'search_control': 'random', 'seed': 7, 'step_logging_interval': 10}
    
    # data =  np.full((1, 4), 1)
    # save_data_zodb(test_config, data)
    # save_data_zodb(test_config, data)
    # save_data_zodb(test_config, data)
    # save_data_zodb(test_config, data)
    # save_data_zodb(test_config, data)
    # save_data_zodb(test_config, data)
    # save_data_zodb(test_config, data)
    # save_data_zodb(test_config, data)
    # save_data_zodb(test_config, data)
    # save_data_zodb(test_config, data)
    # save_data_zodb(test_config, data)
    # save_data_zodb(test_config, data)
    # save_data_zodb(test_config, data)
    # save_data_zodb(test_config, data)
    # save_data_zodb(test_config, data)
    code.interact()

    