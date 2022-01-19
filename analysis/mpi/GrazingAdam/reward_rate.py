import os
import sys

from numpy.lib.function_base import average
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
from src.analysis.plot_utils import load_experiment_data, get_json_handle, get_x_range
import argparse
from src.utils import run_utils
import importlib.util
import matplotlib

def generatePlot(data):
    
    # print("backend", plt.rcParams["backend"])
    # For now, we can't use the default MacOSX backend since it will give me terrible corruptions
    matplotlib.use("TkAgg")
    # print(return_data)
    # Processing data here so the dimensions are correct
    print(data.keys())
    reward_rate = data["reward_rate"]
    reward_rate = np.array(reward_rate)
    reward_rate = reward_rate[0, :]

    max_reward_rate = data["max_reward_rate"]
    max_reward_rate = np.array(max_reward_rate)
    print(max_reward_rate.shape)
    max_reward_rate = max_reward_rate[0, :]

    # Getting file name
    save_file = prompt_user_for_file_name('./plots/', 'reward_rate_', '', 'png', timestamp=True)
    print(f'Plot will be saved in {save_file}')

    def smooth_arr(arr, step):
        
        smooth_array = np.zeros(arr.shape[0] // step)
        start_index = 0
        
        for i in range(0, arr.shape[0], step):
            end_index = min(i + step, arr.shape[0])
            print(i ,end_index)
            smooth_array[i // step] = np.average(arr[i: end_index])
        return smooth_array

    average_step = 100
    smoothed_reward_rate = smooth_arr(reward_rate, average_step)
    smoothed_max_reward_rate = smooth_arr(max_reward_rate, average_step)
    x_range = get_x_range(0, reward_rate.shape[0] // average_step, average_step)
    print(len(list(x_range)))
    print(smoothed_reward_rate.shape)
    print(smoothed_max_reward_rate.shape)

    plt.figure()
    plt.plot(x_range, smoothed_reward_rate, label='reward_rate')

    plt.plot(x_range, smoothed_max_reward_rate, label='max_reward_rate')
    plt.legend()

    plt.savefig(f'{save_file}', dpi = 300)


if __name__ == "__main__":

    # Parsing arguments
    parser = argparse.ArgumentParser(description='Produces the action values video for GrazingWorld Adam')
    parser.add_argument('parameter_path', help='path to the Python parameter file that contains a get_configuration_list function that returns a list of parameters to run')
    args = parser.parse_args()

    parameter_path = args.parameter_path

    # Getting parameter list from parameter_path
    param_spec = importlib.util.spec_from_file_location("ParamModule", parameter_path)
    ParamModule = importlib.util.module_from_spec(param_spec)
    param_spec.loader.exec_module(ParamModule)
    parameter_list = ParamModule.get_configuration_list()

    data = run_utils.load_data(parameter_list[0])

    generatePlot(data)

    exit()