import os
import sys
sys.path.append(os.getcwd())

from experiment_utils.sweep_configs.common import get_configuration_list_from_file_path
from experiment_utils.data_io.configs import check_config_completed, get_folder_name, DB_FOLDER, load_all_db_configs_and_keys, load_data_from_config_zodb
from experiment_utils.data_io.io.zodb_io import BatchDBAccess
from src.utils.run_utils import experiment_completed

# get_configuration_list function is required for 
def get_configuration_list():
    root = 'experiments/chunlok/env_tmaze/'
    files = [
        'baseline.py',
        # 'skip_opt_option.py',
        # 'skip_optimal.py',
        'skip.py',
        # 'skip_opt_option_learn_skip.py',
    ]
    
    parameter_list = [param  for file in files for param in get_configuration_list_from_file_path(root + file)]

    # return parameter_list

    # Assuming that they all go in the same folder
    db_folder = get_folder_name(parameter_list[0], DB_FOLDER)

    def incomplete_filter(config):
        return not check_config_completed(config)

    # Assuming all configs belong to the same folder so we aren't opening/closing the DB over and over again to check each parameter
    with BatchDBAccess(db_folder):
        incomplete_parameter_list = list(filter(incomplete_filter, parameter_list))

    return incomplete_parameter_list