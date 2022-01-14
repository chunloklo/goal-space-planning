# [chunlok-2022-01-11] Common functions that doesn't have anywhere else to go 
# If this file gets too big, then it's probably time to separate things out.

import importlib.util
import argparse
from typing import Any, Callable, Optional

def get_parameter_list_from_file_path(file_path: str):
    # [chunlok-2022-01-11] This little duplicate code here is likely okay for now (similar code in get_run_function_from_file_path) 
    # Getting parameter list from parameter_path
    param_spec = importlib.util.spec_from_file_location("ParamModule", file_path)
    ParamModule = importlib.util.module_from_spec(param_spec)
    param_spec.loader.exec_module(ParamModule)
    parameter_list = ParamModule.get_parameter_list()
    return parameter_list

def get_run_function_from_file_path(file_path: str):
    # [chunlok-2022-01-11] This little duplicate code here is likely okay for now (similar code in get_parameter_list_from_file_path) 
    # Getting run function from run_path
    run_spec = importlib.util.spec_from_file_location("RunModule", file_path)
    RunModule = importlib.util.module_from_spec(run_spec)
    run_spec.loader.exec_module(RunModule)
    run_func = RunModule.run
    return run_func

def get_aux_config_from_file_path(file_path: Optional[str]):
    if file_path is None:
        return None
    run_spec = importlib.util.spec_from_file_location("AuxConfigModule", file_path)
    AuxConfigModule = importlib.util.module_from_spec(run_spec)
    run_spec.loader.exec_module(AuxConfigModule)
    aux_config = AuxConfigModule.get_auxiliary_configuration()
    return aux_config

def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Adds positional run_path, parameter_path, and auxiliary_config_path to the parser.

    Args:
        parser (argparse.ArgumentParser): ArgumentParser to add args to

    Returns:
        [argparse.ArgumentParser]: The parser with which the arguments have been added
    """
    parser.add_argument('run_path', help='Path to the Python run file that contains a run(parameter: dict) function that runs the experiment with the specified parameters')
    parser.add_argument('parameter_path', help='Path to the Python parameter file that contains a get_parameter_list function that returns a list of parameters to run')
    parser.add_argument('aux_config_path', nargs='?', help='Path to the Python auxiliary config file that contains a get_parameter_list function that returns a list of parameters to run')
    return parser

def run_with_optional_aux_config(run_func: Callable, config: Any, aux_config: Any):
    
    if aux_config is not None:
        run_func(config, aux_config)
    else:
        run_func(config)
