
import os
import sys
sys.path.append(os.getcwd())

from typing import Any, Callable, Dict, List, Tuple
import argparse
from src.utils import run_utils
import importlib.util
from copy import copy
from  experiment_utils.analysis_common.configs import group_configs
import numpy as np
from pprint import pprint
import matplotlib
from src.analysis.gridworld_utils import _get_corner_loc, _get_q_value_patch_paths, get_text_location, prompt_user_for_file_name, get_file_name, scale_value, _plot_init, prompt_episode_display_range
from src.analysis.plot_utils import load_experiment_data, get_json_handle, get_x_range
import matplotlib.pyplot as plt
from experiment_utils.sweep_configs.common import get_configuration_list_from_file_path

from src.analysis.plot_utils import load_configuration_list_data
from  experiment_utils.analysis_common.process_line import get_mean_std, mean_chunk_data
from analysis.common import get_best_grouped_param, load_data, load_reward_rate, load_max_reward_rate
from  experiment_utils.analysis_common.cache import cache_local_file
from pathlib import Path

STEP_SIZE = 10

# Plots the reward rate for a single run. Mainly used for debugging

def plot_single_reward_rate(ax, param_file_name: str, label: str=None, key=None):
    if label is None:
        label = Path(param_file_name).stem

    parameter_list = get_configuration_list_from_file_path(param_file_name)
    # print("plotting only the first index of the config list")
    index = 0

    ############ STANDARD
    data = load_data(parameter_list[index], key)

    print(data.shape)
    run_data = mean_chunk_data(data, STEP_SIZE, 0)
    # print(run_data.shape)

    x_range = get_x_range(0, run_data.shape[0], STEP_SIZE)

    # print(len(list(x_range)))
    ax.plot(x_range, run_data, label=label)

    ####### Individual skip probability weights
    # data = load_data(parameter_list[index], 'skip_probability_weights')
    # print(data.shape)
    

    # for i in range(0, 3840, 196):
    #     plt.axvline(x=i)

    # for i in [0, 7, 14, 7 + 15]:
    #     plot_data = data[:, i]
    #     print(data.shape)
    #     run_data = mean_chunk_data(plot_data, STEP_SIZE, 0)
    #     # print(run_data.shape)

    #     # Accumulating
    #     # for i in range(1, len(run_data)):
    #     #     run_data[i] = run_data[i] + run_data[i - 1]

    #     x_range = get_x_range(0, run_data.shape[0], STEP_SIZE)

    #     # print(len(list(x_range)))
    #     ax.plot(x_range, run_data, label=i)

    
def plot_max_reward_rate(ax, param_file_name: str):
    parameter_list = get_configuration_list_from_file_path(param_file_name)
    parameter_list = [parameter_list[0]]
    
    # doesn't matter what we do here.
    max_reward_rate = load_data(parameter_list[0], 'max_reward_rate')

    print(max_reward_rate.shape)
    max_reward_rate = mean_chunk_data(max_reward_rate, STEP_SIZE, 0)
    x_range = get_x_range(0, max_reward_rate.shape[0], STEP_SIZE)
    

    ax.plot(x_range, max_reward_rate, label='max reward rate')

    # fixed_max_reward_rate = [-0.05384615384] * len(list(x_range))
    # fixed_lower_max_reward_rate = [-0.0888888889]* len(list(x_range))

    # fixed_max_reward_rate = [-0.0888888889] * 799 + [-0.05384615384] * 799
    # ax.plot(x_range, max_reward_rate, label='max reward rate')
    # ax.plot(x_range, fixed_max_reward_rate, label='fixed max reward rate')
    # ax.plot(fixed_max_reward_rate, label='max reward rate')

def create_individual_plot(ax, file_name):
    plot_max_reward_rate(ax, file_name)
    plot_single_reward_rate(ax, file_name, None)      


if __name__ == "__main__":
    # create_individual_plot('experiments/chunlok/mpi/extended/collective/dyna_gpi_only_low_init_sweep.py', 'okay')

    # parameter_path = 'experiments/chunlok/mpi/extended_half/collective/dyna_ourgpi_maxaction.py'
    fig, ax = plt.subplots(figsize=(10, 6))

    key = 'policy_loss'
    # plot_max_reward_rate(ax, 'experiments/chunlok/env_tmaze/baseline.py')
    plot_single_reward_rate(ax, 'experiments/pinball/GSP_learn_state_to_goal_est.py', key=key)
    # plot_single_reward_rate(ax, 'experiments/pinball/GSP_learning.py', key=key)
    # plot_single_reward_rate(ax, 'experiments/chunlok/env_tmaze/skip.py')
    # plot_single_reward_rate(ax, 'experiments/chunlok/env_tmaze/skip_optimal.py')
    # plot_single_reward_rate(ax, 'experiments/chunlok/env_tmaze/skip_opt_option.py')

    plt.legend()
    # plt.title(alg_name)

    # ax.set_xlim([600, 1200])

    # Getting file name
    save_file = get_file_name('./plots/', f'{key}', 'png', timestamp=True)
    # print(f'Plot will be saved in {save_file}')
    plt.savefig(f'{save_file}', dpi = 300)
    
    # plt.show()

        