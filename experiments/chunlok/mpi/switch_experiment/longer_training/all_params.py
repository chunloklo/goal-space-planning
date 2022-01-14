import os
import sys
sys.path.append(os.getcwd())

from ParameterSweep.common import get_parameter_list_from_file_path
from src.utils.run_utils import experiment_completed

# get_parameter_list function is required for 
def get_parameter_list():
    files = [
        'experiments/chunlok/mpi/switch_experiment/collective/dyna_backgroundgpi_only_sweep.py',
        'experiments/chunlok/mpi/switch_experiment/collective/dyna_gpi_only_sweep.py',
        'experiments/chunlok/mpi/switch_experiment/collective/dyna_sweep.py',
        'experiments/chunlok/mpi/switch_experiment/collective/dyna_backgroundgpi_sweep.py',
        'experiments/chunlok/mpi/switch_experiment/collective/dyna_gpi_sweep.py',
        'experiments/chunlok/mpi/switch_experiment/collective/dynaoptions_sweep.py',
    ]
    
    parameter_list = [param  for file in files for param in get_parameter_list_from_file_path(file)]
    incomplete_parameter_list = list(filter(lambda param: not experiment_completed(param, pushup=False), parameter_list))

    return incomplete_parameter_list