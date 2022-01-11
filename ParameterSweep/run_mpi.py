from mpi4py import MPI
from mpi4py.util import dtlib
import numpy as np
import sys
import importlib.util
import argparse

# Parsing arguments
parser = argparse.ArgumentParser(description='MPI file that is ran on each task that is spawned through mpiexec or similar functions')
parser.add_argument('parameter_path', help='path to the Python parameter file that contains a get_parameter_list function that returns a list of parameters to run')
parser.add_argument('run_path', help='path to the Python run file that contains a run(parameter: dict) function that runs the experiment with the specified parameters')
args = parser.parse_args()

parameter_path = args.parameter_path
run_path = args.run_path

# Getting parameter list from parameter_path
param_spec = importlib.util.spec_from_file_location("ParamModule", parameter_path)
ParamModule = importlib.util.module_from_spec(param_spec)
param_spec.loader.exec_module(ParamModule)
parameter_list = ParamModule.get_parameter_list()

# Getting run function from run_path
run_spec = importlib.util.spec_from_file_location("RunModule", run_path)
RunModule = importlib.util.module_from_spec(run_spec)
run_spec.loader.exec_module(RunModule)
run_fuc = RunModule.run

# print(parameter_list)
print(len(parameter_list))

# Standard MPI info
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = MPI.COMM_WORLD.Get_size()
name = MPI.Get_processor_name()

# Creating a counter for each task to grab tasks off of
# The counter will represent the next parameter that needs to be ran,
# Each task, when waiting, will grab the top available number and run it.
# This is repeated until the entire parameter list is ran

datatype = MPI.UINT64_T
np_dtype = dtlib.to_numpy_dtype(datatype)
itemsize = datatype.Get_size()

# Creating window
win_size = itemsize if rank == 0 else 1
win = MPI.Win.Allocate(win_size, comm=comm)

# # Creating buffer for each task
buf = np.empty(1, dtype=np_dtype)

# Starting the counter at 0 
if rank == 0:
    buf.fill(0)
    win.Lock(rank=0)
    win.Put(buf, target_rank=0)
    win.Unlock(rank=0)
comm.Barrier()

next_available_param_index = 0
num_parameters = len(parameter_list)

while next_available_param_index < num_parameters:
    win.Lock(rank=0)
    win.Get(buf, target_rank=0)
    next_available_param_index = np.copy(buf)

    buf += 1
    win.Put(buf , target_rank=0)
    win.Unlock(rank=0)

    # if next_available_param_index > 0: break
    if next_available_param_index >= num_parameters:
        # You're done!
        break

    sys.stdout.write(
        f"Running parameter {next_available_param_index}! I am process {rank} of {size} on {name}\n")





