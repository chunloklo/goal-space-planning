import math
from typing import Any, Callable, Dict, List
from copy import copy
from tqdm import tqdm
import numpy as np

def get_key_data(data, key):
    key_data = []
    for d in data:
        key_data.append(d[key])

    return key_data

def trim_1d_array(data, percentage):
    start_index = math.floor(percentage[0] * len(data))
    end_index = math.ceil(percentage[1] * len(data))
    return data[start_index : end_index]

class SweepInfo:

    def __init__(self, sweep_configs: List[Dict]):
        
        self.filters = {}
        self.configs = copy(sweep_configs)
        self.diff_list = {}

        self._update_diff_list()

    def _update_diff_list(self):
        diff_list = {}
        base_config = None

        filtered_configs = self._filter_configs()
        for config in filtered_configs:

            if base_config is None:
                base_config = config

            for key in config.keys():
                if base_config[key] != config[key]:
                    if key not in diff_list.keys():
                        diff_list[key] = [base_config[key]]

                    if config[key] not in diff_list[key]:
                        diff_list[key].append(config[key])
        self.diff_list = diff_list

    def add_filter(self, name, value):
        self.filters[name] = value
        self._update_diff_list()

    def remove_filter(self, name):
        self.filters.pop(name)
        self._update_diff_list()
    
    def clear_filter(self):
        self.filters = {}
        self._update_diff_list()
    
    def set_filter(self, filter):
        self.filters = filter
        self._update_diff_list()

    def _filter_configs(self):
        def filter_func(config):
            for key, value in self.filters.items():
                if config[key] != value:
                    return False
            return True

        filtered_configs = [config for config in self.configs if filter_func(config)]

        return filtered_configs
    
def group_configs(parameter_list, ignore_keys: List[str]):
    bin_base_configs = []
    bin_configs = []

    configs = parameter_list

    #  Check whether parameters are equal given ignored keys
    def param_equal(a: Dict, b: Dict):
        for k, v in a.items():
            if k in ignore_keys:
                continue
            if k not in b or b[k] != v:
                return False
        return True


    for config in configs:
        # Attempt to group into bins
        placed = False

        # This is a manual way of doing this... O(N^2)?
        for ind, bin_base_config in enumerate(bin_base_configs):
            # If it found an equal config, add it in.
            if param_equal(bin_base_config, config):
                # merge into bins
                bin_configs[ind].append(config)
                placed = True
                break
        
        if not placed:
            bin_configs.append([config])
            # Copy base config without the ignored keys
            base_config = {k: v for k, v in config.items() if k not in ignore_keys}
            bin_base_configs.append(base_config)

    return [(base_config, bin_configs[i]) for i, base_config in enumerate(bin_base_configs)]