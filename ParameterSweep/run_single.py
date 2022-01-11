from mpi4py import MPI
from mpi4py.util import dtlib
import numpy as np
import sys
import importlib.util
import argparse

# Parsing arguments
parser = argparse.ArgumentParser(description='MPI file that is ran on each task that is spawned through mpiexec or similar functions')
parser.add_argument('parameter_path', help='Path to the Python parameter file that contains a get_parameter_list function that returns a list of parameters to run')
parser.add_argument('run_path', help='Path to the Python run file that contains a run(parameter: dict) function that runs the experiment with the specified parameters')
parser.add_argument('index', type = int, help='Index on the parameter list to run')
args = parser.parse_args()

parameter_path = args.parameter_path
run_path = args.run_path
index = args.index

# Getting parameter list from parameter_path
param_spec = importlib.util.spec_from_file_location("ParamModule", parameter_path)
ParamModule = importlib.util.module_from_spec(param_spec)
param_spec.loader.exec_module(ParamModule)
parameter_list = ParamModule.get_parameter_list()

# Getting run function from run_path
run_spec = importlib.util.spec_from_file_location("RunModule", run_path)
RunModule = importlib.util.module_from_spec(run_spec)
run_spec.loader.exec_module(RunModule)
run_func = RunModule.run

run_func(parameter_list[index], True)