# sweep_configs
This is a small library for running many independent experiments to sweep over different configurations. It's designed to do just that, and nothing more.

## Overview
This library's run files takes in paths to multiple python files in order to make this happen. For each collection of experiments that you want to run, you need the following 2 things:

1. A python file that contains a `get_configuration_list` function that returns a list of configurations (in the form of a list) that you want to run.
2. A python file that contains a `run(config)` function that runs your experiment with a specific configuration.

`run_mpi.py` is designed to be ran with MPI and will run all configurations returned from `get_configuration_list` in parallel in the number of nodes available.

Example command:
```mpiexec -n 8 python run_mpi.py <configuration_list_file> <run_function_file>```

`run_single.py` is designed to be a debugging file that allows you to run a single configuration from the configuration list before using `run_mpi.py`. It allows you to specify exactly which index you want to run from your configuration list.

Example command:
```python run_single.py <configuration_list_file> <run_function_file> <index>```

## Auxiliary configurations
On occasions, especially with your debug experiments, you might want to run your experiment with a set of auxiliary parameters that doesn't affect the experiment results. For example, you might want to show debug information, progress, or print-outs on local runs of your experiment, or you might want debug information from your sweep experiments. Auxiliary configurations allows you to provide a set of additional parameters to your experiment for these cases. 

**The intention for the auxiliary configurations is for configs that will NOT affect the experiment results.** But if course you can do what you want, just my advice.

To add an auxiliary configuration, create a python file with a function named `get_auxiliary_configuration`, and provide the path to the file when running as the third positional argument.

Example `python run_single.py <configuration_list_file> <run_function_file> <aux_config_file> <index>`

Then, you need to modify `run` in your run file such that it accepts arguments in the form of `run(config, aux_config=<Default>)`.

## Example
To see how to implement your own sweep, check out files under `sweep_configs/example/`. Run `dummy_bash.sh` to see an example output.

## Performing grid search
A common way to perform sweeps is through grid-search. `generate_configs.get_sorted_configuration_list_from_dict` is designed to generate a configuration list just for that scenario. It generates a sorted list of configurations from a dictionary describing what parameter values you want to sweep over.

## IMPORTANT: Deterministic configuration list 
It is imperative that the order of your configuration list is deterministic. When running with MPI, a copy of the configuration list will be created by each task and each task gets assigned configurations to run based on its index in the list. If the list is not deterministic, there is no guarantees that all configurations in the list will be ran.

## Running on Compute Canada
Example scripts on how to use this library to run configuration sweeps can be found in `sweep_configs/compute_canada`.

