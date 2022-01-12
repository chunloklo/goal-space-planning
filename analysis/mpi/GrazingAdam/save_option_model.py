import os
import sys
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


import argparse
from src.utils import run_utils
import importlib.util
import matplotlib


COLUMN_MAX = 12
ROW_MAX = 8

def generatePlot(data):
    # print(data["model_r"].shape)

    # print(return_data)
    # Processing data here so the dimensions are correct
    np.set_printoptions(threshold=sys.maxsize)
    # print(data.keys())
    # print(np.array(data["model_r"])[0, -1])
    # sds
    # print(data["model_r"][0, 0, -1, :, :])
    # print(data["model_discount"][0, 0, -1, :, :])
    # print(data["model_transition"][0, 0, -1, :, :, :])
    # asdas

    np.save('src/environments/data/GrazingWorldAdam_OptionModel_r.npy', np.array(data["model_r"])[0, -1], False)
    np.save('src/environments/data/GrazingWorldAdam_OptionModel_discount.npy', np.array(data["model_discount"])[0, -1], False)
    np.save('src/environments/data/GrazingWorldAdam_OptionModel_transition.npy', np.array(data["model_transition"])[0, -1], False)
    

    np.save('src/environments/data/GrazingWorldAdam_ActionOptionModel_r.npy', np.array(data["action_model_r"])[0, -1], False)
    np.save('src/environments/data/GrazingWorldAdam_ActionOptionModel_discount.npy', np.array(data["action_model_discount"])[0, -1], False)
    np.save('src/environments/data/GrazingWorldAdam_ActionOptionModel_transition.npy', np.array(data["action_model_transition"])[0, -1], False)
    


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

    # Parsing arguments
    parser = argparse.ArgumentParser(description='Produces the action values video for GrazingWorld Adam')
    parser.add_argument('parameter_path', help='path to the Python parameter file that contains a get_parameter_list function that returns a list of parameters to run')
    args = parser.parse_args()

    parameter_path = args.parameter_path

    # Getting parameter list from parameter_path
    param_spec = importlib.util.spec_from_file_location("ParamModule", parameter_path)
    ParamModule = importlib.util.module_from_spec(param_spec)
    param_spec.loader.exec_module(ParamModule)
    parameter_list = ParamModule.get_parameter_list()

    data = run_utils.load_data(parameter_list[0])

    generatePlot(data)

    exit()