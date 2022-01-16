import sys
from src.utils.json_handling import get_sorted_dict , get_param_iterable_runs
from src.utils import analysis_utils
import numpy as np 
from typing import Callable, Dict

def get_json_handle():
    json_files = sys.argv[1:] # all the json files
    # convert all json files to dict
    json_handles = [get_sorted_dict(j) for j in json_files]

    # Logic for choosing which json handle
    print("grabbing only the first experiment for visualization")
    return json_handles[0]

def load_experiment_data(json_handle, load_keys: list = None):
    # if load_keys is None, then it loads all the keys
    iterables = get_param_iterable_runs(json_handle)
        
    for i in iterables:
        print(i)
        return_data = analysis_utils.load_different_runs_all_data(i, load_keys)
        print(return_data.keys())
        # mean_return_data, stderr_return_data = process_runs(return_data)
        pass

    # Messy right now, but its okay
    return return_data

def get_x_range(start: int, num_logs: int, interval: int):
    return range(start, start + num_logs * interval, interval)

def window_smoothing(data: np.array, window_size: int):
    smoothed_data = np.zeros(data.shape)
    # print(data.shape)

    for i in range(data.shape[0]):
        start_index = min(i, data.shape[0] - window_size)
        end_index =  min(data.shape[0], start_index + window_size)
        # print(start_index, end_index)
        smoothed_data[i] = np.mean(data[start_index: end_index])
    # sadas
    return smoothed_data

def load_configuration_list_data(parameter_list: list[Dict], load_func: Callable):
    data = [load_func(param) for param in parameter_list]
    return data