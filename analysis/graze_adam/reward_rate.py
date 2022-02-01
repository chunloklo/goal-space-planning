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
from src.analysis.plot_utils import load_experiment_data, get_json_handle, get_x_range

def generatePlot(json_handle):
    data = load_experiment_data(json_handle)

    # print(return_data)
    # Processing data here so the dimensions are correct
    print(data.keys())
    reward_rate = data["reward_rate"]
    print(reward_rate.shape)
    reward_rate = reward_rate[0, 0, :]

    max_reward_rate = data["max_reward_rate"]
    print(max_reward_rate.shape)
    max_reward_rate = max_reward_rate[0, 0, :]

    # Getting file name
    save_file = prompt_user_for_file_name('./plots/', 'reward_rate_', '', 'png', timestamp=True)
    print(f'Plot will be saved in {save_file}')

    x_range = get_x_range(0, reward_rate.shape[0], 1)

    plt.figure()
    plt.plot(x_range, reward_rate, label='reward_rate')

    plt.plot(x_range, max_reward_rate, label='max_reward_rate')
    plt.legend()

    plt.savefig(f'{save_file}', dpi = 300)

if __name__ == "__main__":

    # read the arguments etc
    if len(sys.argv) < 2:
        print("usage : python analysis/process_data.py <list of json files")
        exit()


    json_handle = get_json_handle()

    # Only use the first handle for now?
    generatePlot(json_handle)

    exit()