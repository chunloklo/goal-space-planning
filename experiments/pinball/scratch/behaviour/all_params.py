import os
import sys
sys.path.append(os.getcwd())

from experiment_utils.sweep_configs.common import get_configuration_list_from_file_path
from experiment_utils.data_io.configs import check_config_completed, get_folder_name, DB_FOLDER, get_incomplete_configuration_list, load_all_db_configs_and_keys, load_data_from_config_zodb
from experiment_utils.data_io.io.zodb_io import BatchDBAccess
from src.utils.run_utils import experiment_completed

# get_configuration_list function is required for 
def get_configuration_list():
    root = 'experiments/pinball/scratch/behaviour/'
    files = [
        # 'baseline.py'
        # 'short_scratch_model_gsp_learn_10k.py',
        # 'short_scratch_model_gsp_learn_25k.py',
        # 'short_scratch_model_gsp_learn_50k.py',
        # 'short_scratch_model_gsp_learn_75k.py',
        'short_scratch_model_gsp_learn_100k.py',
        'short_scratch_model_gsp_learn_125k.py',
        'short_scratch_model_gsp_learn_150k.py',
        'short_scratch_model_gsp_learn_175k.py',
        'short_scratch_model_gsp_learn_final.py',
    ]
    
    parameter_list = [param  for file in files for param in get_configuration_list_from_file_path(root + file)]
    # return parameter_list
    incomplete_parameter_list = get_incomplete_configuration_list(parameter_list)
    return incomplete_parameter_list