from mpi4py import MPI
from mpi4py.util import dtlib
import numpy as np
import sys
import importlib.util
import argparse

from common import get_parameter_list_from_file_path, get_run_function_from_file_path, add_common_args, get_aux_config_from_file_path, run_with_optional_aux_config

# Mainly used for debugging and for testing whether your run is successful before running multiple runs in parallel with mpi

# Parsing arguments
parser = argparse.ArgumentParser(description='MPI file that is ran on each task that is spawned through mpiexec or similar functions')
parser = add_common_args(parser)
parser.add_argument('index', type = int, help='Index on the parameter list to run')
args = parser.parse_args()

parameter_path = args.parameter_path
run_path = args.run_path
aux_config_path = args.aux_config_path
index = args.index

# Getting parameter list from parameter_path
parameter_list = get_parameter_list_from_file_path(parameter_path)

# Getting run function from run_path
run_func = get_run_function_from_file_path(run_path)

# Getting auxiliary config from auxiliary config path
aux_config = get_aux_config_from_file_path(aux_config_path)

run_with_optional_aux_config(parameter_list[index], aux_config)