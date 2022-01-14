#!/bin/sh
#SBATCH --account=rrg-whitem
#SBATCH --time=00:00:30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem-per-cpu=1024M
#SBATCH --mail-user=chunlok@ualberta.ca
#SBATCH --mail-type=ALL
source ~/env-38/bin/activate
cd /project/6010404/chunlok/dyna-options
mpiexec python ParameterSweep/impl_test/queue_test.py