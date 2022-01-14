#!/bin/sh
#SBATCH --account=rrg-whitem
#SBATCH --time=00:5:00
#SBATCH --ntasks=33
#SBATCH --mem-per-cpu=4000M
#SBATCH --mail-user=chunlok@ualberta.ca
#SBATCH --mail-type=ALL
source ~/env-38/bin/activate
cd /project/6010404/chunlok/dyna-options
mpiexec python ParameterSweep/run_mpi.py experiments/chunlok/mpi/dyna_sweep.py src/mpi_run.py 