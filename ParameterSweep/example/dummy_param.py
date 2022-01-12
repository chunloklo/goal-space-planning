import os
import sys
sys.path.append(os.getcwd())

from ParameterSweep.parameters import get_sorted_parameter_list_from_dict

# get_parameter_list function is required for 
def get_parameter_list():
    parameter_dict = {
        "seed": list(range(0, 10)),
        "param0": [100, 0, 55]
    }

    parameter_list = get_sorted_parameter_list_from_dict(parameter_dict)
    return parameter_list