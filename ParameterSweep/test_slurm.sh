#!/bin/sh
#SBATCH --account=rrg-whitem
#SBATCH --time=00:00:30
#SBATCH --ntasks=64
#SBATCH --mem-per-cpu=256M
#SBATCH --mail-user=chunlok@ualberta.ca
#SBATCH --mail-type=ALL
source ~/env-38/bin/activate
cd /project/6010404/chunlok/dyna-options
mpirun python ParameterSweep/slurm_count.py