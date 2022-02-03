import os
import sys
sys.path.append(os.getcwd())

from experiment_utils.sweep_configs.common import get_configuration_list_from_file_path
from src.utils.run_utils import experiment_completed

# get_configuration_list function is required for 
def get_configuration_list():
    root = 'experiments/chunlok/graze_learn/'
    files = [
        'dyna.py',
        'dynaoptions.py',
        'OCG.py',
        'OCI_action.py',
        'OCI.py'
    ]
    
    parameter_list = [param  for file in files for param in get_configuration_list_from_file_path(root + file)]
    return parameter_list
    # incomplete_parameter_list = list(filter(lambda param: not experiment_completed(param, pushup=False), parameter_list))
    # print(len(incomplete_parameter_list))
    # return incomplete_parameter_list