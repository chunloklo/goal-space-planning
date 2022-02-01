import os
import sys

from experiment_utils.sweep_configs.common import get_configuration_list_from_file_path
sys.path.append(os.getcwd())

from genericpath import isdir
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib import cm
from matplotlib.animation import FuncAnimation


from src.analysis.learning_curve import plotBest
from src.experiment import ExperimentModel
from PyExpUtils.results.results import loadResults, whereParameterGreaterEq, whereParametersEqual, find
from PyExpUtils.utils.arrays import first
from tqdm import tqdm

# Mass taking imports from process_data for now
import os, sys, time, copy
sys.path.append(os.getcwd())
import numpy as np

from src.utils.json_handling import get_sorted_dict , get_param_iterable_runs
from src.utils.formatting import create_file_name, create_folder
from src.utils import analysis_utils
from src.analysis.gridworld_utils import _get_corner_loc, _get_q_value_patch_paths, get_text_location, prompt_user_for_file_name, get_action_offset, scale_value, _plot_init, prompt_episode_display_range
from src.analysis.plot_utils import get_x_range

COLUMN_MAX = 12
ROW_MAX = 8

def generatePlot(json_handle):
    data = load_experiment_data(json_handle)

    # print(return_data)
    # Processing data here so the dimensions are correct
    np.set_printoptions(threshold=sys.maxsize)
    # print(data.keys())
    # print(data["model_r"].shape)
    # print(data["model_r"][0, 0, -1, :, :])
    # print(data["model_discount"][0, 0, -1, :, :])
    # print(data["model_transition"][0, 0, -1, :, :, :])

    np.save('src/environments/data/GrazingWorldAdam_OptionModel_r.npy', data["model_r"][0, 0, -1], False)
    np.save('src/environments/data/GrazingWorldAdam_OptionModel_discount.npy', data["model_discount"][0, 0, -1], False)
    np.save('src/environments/data/GrazingWorldAdam_OptionModel_transition.npy', data["model_transition"][0, 0, -1], False)
    

    np.save('src/environments/data/GrazingWorldAdam_ActionOptionModel_r.npy', data["action_model_r"][0, 0, -1], False)
    np.save('src/environments/data/GrazingWorldAdam_ActionOptionModel_discount.npy', data["action_model_discount"][0, 0, -1], False)
    np.save('src/environments/data/GrazingWorldAdam_ActionOptionModel_transition.npy', data["action_model_transition"][0, 0, -1], False)
    


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

if __name__ == "__main__":

    param_file_name = 'experiments/chunlok/pretrain_model/graze_adam/dyna_config.py'
    parameter_list = get_configuration_list_from_file_path(param_file_name)

    # Only use the first handle for now?
    generatePlot(parameter_list)

    exit()