folder=`dirname "$0"` # Grabbing the base folder of this file so this runs from any directory
mpiexec -n 4 python $folder/../run_mpi.py $folder/dummy_run.py $folder/dummy_configs.py $folder/dummy_aux_config.py