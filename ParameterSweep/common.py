# [chunlok-2022-01-11] Common functions that doesn't have anywhere else to go 
# If this file gets too big, then it's probably time to separate things out.

import importlib.util

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