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
from src.environments.HMaze import HMaze

from experiment_utils.sweep_configs.common import get_configuration_list_from_file_path
from src.analysis.plot_utils import load_configuration_list_data
from analysis.common import get_best_grouped_param, load_reward_rate, load_max_reward_rate, plot_mean_ribbon
from  experiment_utils.analysis_common.configs import group_configs
from analysis.common import load_data
import matplotlib
from src.analysis.gridworld_utils import _get_corner_loc, _get_q_value_patch_paths, get_text_location, prompt_user_for_file_name, get_file_name, scale_value, _plot_init, prompt_episode_display_range

def create_plot(data, key):
    plt.legend()
    plt.title(key)

    fig, axes = plt.subplots(4, 4, figsize=(32, 32))
    # print(axes)
    axes = axes.flatten()

    num_goals = 16
    for g in range(num_goals):
        axes[g].plot(data[:, g, :], label=list(range(num_goals)))
        axes[g].set_title(g)
        axes[g].legend()

        if g == 0:
            print(data[-1, g, :])

     # ax.set_xlim([600, 1200])

    
    # Getting file name
    save_file = get_file_name('./plots/', f'{key}', 'png', timestamp=True)

    # print(f'Plot will be saved in {save_file}')
    plt.savefig(f'{save_file}', dpi = 300)

if __name__ == "__main__":
    parameter_path = 'experiments/pinball/GSP_impl_test.py'
    parameter_list = get_configuration_list_from_file_path(parameter_path)
    
    key = 'goal_init'
    data = load_data(parameter_list[0], key)

    # print(data[-1, 0])
    # asda

    create_plot(data, key)
    exit()