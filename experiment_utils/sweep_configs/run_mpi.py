# This is require to turn threading off before MPI is initialized
import mpi4py
mpi4py.rc.thread_level = 'single' # or perhaps 'serialized'
mpi4py.rc.threads = False

# This import statement also initializes MPI
from mpi4py import MPI

from mpi4py.util import dtlib
import argparse
import numpy as np
import sys

from common import get_configuration_list_from_file_path, get_run_function_from_file_path, add_common_args, get_aux_config_from_file_path, run_with_optional_aux_config

# Parsing arguments
parser = argparse.ArgumentParser(description='MPI file that is ran on each task that is spawned through mpiexec or similar functions')
parser = add_common_args(parser)
args = parser.parse_args()

configuration_path = args.config_path
run_path = args.run_path
aux_config_path = args.aux_config_path

# Getting configuration list from parameter_path
configuration_list = get_configuration_list_from_file_path(configuration_path)

# Getting run function from run_path
run_func = get_run_function_from_file_path(run_path)

# Getting auxiliary config from auxiliary config path
aux_config = get_aux_config_from_file_path(aux_config_path)

# Standard MPI info
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = MPI.COMM_WORLD.Get_size()
name = MPI.Get_processor_name()

# Creating a counter for each task to grab tasks off of.
# The counter will represent the next parameter that needs to be ran,
# Each task, when done with their previous job, will grab the top available number and run it.
# This is repeated until the entire parameter list is ran

# This is a little janky storing the counter integer as a 0D np array
# However, this is taken off the mpi4py doc, and debugging on CC is a pain
# Plus it works, so it stays for now.
datatype = MPI.UINT64_T
np_dtype = dtlib.to_numpy_dtype(datatype)
itemsize = datatype.Get_size()

# Creating window for sharing the counter information
win_size = itemsize if rank == 0 else 1
win = MPI.Win.Allocate(win_size, comm=comm)

# Creating buffer and starting the counter at 0 
if rank == 0:
    # Creating buffer for each task
    buf = np.empty(1, dtype=np_dtype)
    buf.fill(0)
    win.Lock(rank=0)
    win.Put(buf, target_rank=0)
    win.Unlock(rank=0)

# Ensures buffer is initialized before allowing processes to start querying it
comm.Barrier()

# buffer for the result and the accumulation amount
accumulate = np.ones(1, dtype=np_dtype)
result = np.zeros(1, dtype=np_dtype)

# The maximum index in the parameter list that needs to be ran:
max_param_index = len(configuration_list) - 1

while result <= max_param_index:
    win.Lock(rank=0)
    # Fetches the counter and increments the counter by 1 (default Op is sum, and accumulate is 1)
    win.Fetch_and_op(accumulate, result, target_rank=0)
    win.Unlock(rank=0)

    # If the top number is greater than the maximum allowed, stop running
    if result > max_param_index:
        break

    # Otherwise, run the function 
    run_with_optional_aux_config(run_func, configuration_list[int(result)], aux_config)

win.Free()
sys.exit(0)

