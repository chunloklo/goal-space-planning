import os
import sys
sys.path.append(os.getcwd())

from experiment_utils.sweep_configs.common import get_configuration_list_from_file_path
from src.utils.run_utils import experiment_completed

# get_configuration_list function is required for 
def get_configuration_list():
    files = [
        'experiments/chunlok/mpi/extended_half/collective/dyna_backgroundgpi_only_low_init_sweep.py',
        'experiments/chunlok/mpi/extended_half/collective/dyna_gpi_only_low_init_sweep.py',
        'experiments/chunlok/mpi/extended_half/collective/dyna_sweep.py',
        'experiments/chunlok/mpi/extended_half/collective/dynaoptions_sweep.py',
        'experiments/chunlok/mpi/extended_half/collective/dyna_ourgpi_maxaction.py',
    ]
    
    parameter_list = [param  for file in files for param in get_configuration_list_from_file_path(file)]
    # return parameter_list
    incomplete_parameter_list = list(filter(lambda param: not experiment_completed(param, pushup=False), parameter_list))
    print(len(incomplete_parameter_list))
    return incomplete_parameter_list