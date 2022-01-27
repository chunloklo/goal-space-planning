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

def create_goal_value_plot(config):
    fig, ax = plt.subplots(figsize=(10, 6))

    data = load_data(config, 'goal_values')

    ax.plot(data)

    plt.legend()
    plt.title('goal values')

    # ax.set_xlim([600, 1200])

    # Getting file name
    save_file = get_file_name('./plots/', f'goal_values', 'png', timestamp=True)


    # print(f'Plot will be saved in {save_file}')
    plt.savefig(f'{save_file}', dpi = 300)

if __name__ == "__main__":

    # Parsing arguments
    parameter_path = 'experiments/chunlok/mpi/hmaze/optionplanning_test.py'
    parameter_list = get_configuration_list_from_file_path(parameter_path)

    create_goal_value_plot(parameter_list[0])
    exit()