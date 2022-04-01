
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
from analysis.common import get_best_grouped_param, load_data, load_reward_rate, load_max_reward_rate, plot_mean_ribbon
from  experiment_utils.analysis_common.cache import cache_local_file
from pathlib import Path

STEP_SIZE = 50

# Plots the reward rate for a single run. Mainly used for debugging

def plot_single_reward_rate(ax, param_file_name: str, label: str=None, color=None):
    if label is None:
        label = Path(param_file_name).stem

    parameter_list = get_configuration_list_from_file_path(param_file_name)
    # print("plotting only the first index of the config list")

    all_data = []
    for param in parameter_list:
        ############ STANDARD
        data = load_data(param, 'reward_rate')
        all_data.append(data)

        print(data.shape)
        run_data = mean_chunk_data(data, STEP_SIZE, 0)
        # print(run_data.shape)

        x_range = get_x_range(0, run_data.shape[0], STEP_SIZE)

        # print(len(list(x_range)))
        ax.plot(x_range, run_data, label=param_file_name, color=color)


def plot_reward_rate_group(ax, group,  color=None):
    all_data = []
    for param in group:
        ############ STANDARD
        data = load_data(param, 'reward_rate')
        all_data.append(data)

        print(data.shape)
        run_data = mean_chunk_data(data, STEP_SIZE, 0)
        # print(run_data.shape)

        x_range = get_x_range(0, run_data.shape[0], STEP_SIZE)

        # print(len(list(x_range)))
        ax.plot(x_range, run_data, color=color)

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

    # ax.set_ylim([0, 10000])
    ax.set_xlabel('episodes')
    ax.set_ylabel('number of steps per episode')
    # plot_max_reward_rate(ax, 'experiments/chunlok/env_tmaze/baseline.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/impl_test.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/GSP_baseline_check.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/test_sweep/GSP_learning_values_more.py', color=None)


    # plot_single_reward_rate(ax, 'experiments/pinball/baseline_sweep/impl_test_batch2_display_0.1.py', color=None)




    param_list = get_configuration_list_from_file_path('experiments/pinball/test_sweep/GSP_learning_values.py')

    groups = group_configs(param_list, ignore_keys=['seed'])

    for group in groups:
        # if group[0]['use_exploration_bonus'] == False:
        #     plot_reward_rate_group(ax, group[1])

        # if group[0]['OCI_update_interval'] == 2 and  group[0]['use_exploration_bonus'] == True and group[0]['polyak_stepsize'] == 0.05:
        #     plot_reward_rate_group(ax, group[1])
        if group[0]['OCI_update_interval'] == 2 and  group[0]['use_exploration_bonus'] == False and group[0]['polyak_stepsize'] == 0.05:
            plot_reward_rate_group(ax, group[1])
        # if group[0]['OCI_update_interval'] == 1 and  group[0]['use_exploration_bonus'] == True and group[0]['polyak_stepsize'] == 0.1:
        #     plot_reward_rate_group(ax, group[1])



    ax.set_ylim([-20, 180])

    # Getting file name
    save_file = get_file_name('./plots/', f'reward_rate', 'png', timestamp=True)
    # print(f'Plot will be saved in {save_file}')
    plt.savefig(f'{save_file}', dpi = 300)
    
    # plt.show()

        