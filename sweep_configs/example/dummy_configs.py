import os
import sys
sys.path.append(os.getcwd())

from sweep_configs.generate_configs import get_sorted_configuration_list_from_dict

# get_configuration_list function is required for 
def get_configuration_list():
    parameter_dict = {
        "seed": list(range(0, 10)),
        "param0": [100, 0, 55]
    }

    parameter_list = get_sorted_configuration_list_from_dict(parameter_dict)
    return parameter_list