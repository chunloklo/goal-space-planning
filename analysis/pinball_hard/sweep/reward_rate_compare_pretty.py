
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
from  experiment_utils.analysis_common.process_line import get_mean_std, get_mean_stderr, mean_chunk_data
from analysis.common import get_best_grouped_param, load_data, load_reward_rate, load_max_reward_rate, plot_mean_ribbon
from  experiment_utils.analysis_common.cache import cache_local_file
from pathlib import Path
from experiment_utils.analysis_common.colors import TOL_BRIGHT
from experiment_utils.data_io.configs import check_config_completed, get_complete_configuration_list, get_folder_name, DB_FOLDER, get_incomplete_configuration_list
from matplotlib import cm

STEP_SIZE = 10

# Plots the reward rate for a single run. Mainly used for debugging

def plot_single_reward_rate(ax, param_file_name: str, label: str=None):
    if label is None:
        label = Path(param_file_name).stem

    parameter_list = get_configuration_list_from_file_path(param_file_name)
    # print("plotting only the first index of the config list")

    all_data = []
    for param in parameter_list:
        ############ STANDARD
        data = load_data(param, 'reward_rate')
        all_data.append(data)

        # print(data.shape)
        # run_data = mean_chunk_data(data, STEP_SIZE, 0)
        # # print(run_data.shape)

        # x_range = get_x_range(0, run_data.shape[0], STEP_SIZE)

        # # print(len(list(x_range)))
        # ax.plot(x_range, run_data, label=label)
    all_data = np.vstack(all_data)

    print(all_data.shape)

    # mean, std = get_mean_stderr(all_data)
    # mean, std = mean_chunk_data(mean, STEP_SIZE), mean_chunk_data(std, STEP_SIZE)

    all_data = mean_chunk_data(all_data, STEP_SIZE)
    mean, std = get_mean_stderr(all_data)
    
    print(std[0:10])
    

    x_range = get_x_range(0, mean.shape[0], STEP_SIZE)

    plot_mean_ribbon(ax, mean, std, x_range, label=param_file_name)

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


def plot_reward_rate_group(ax, group, color=None, postfix='', label=None, linestyle='solid'):
    all_data = []
    for param in group:
        ############ STANDARD
        # print(param)
        data = load_data(param, 'reward_rate')
        all_data.append(data)

        # print(data.shape)
        # run_data = mean_chunk_data(data, STEP_SIZE, 0)
        # # print(run_data.shape)

        # x_range = get_x_range(0, run_data.shape[0], STEP_SIZE)

        # # print(len(list(x_range)))
        # ax.plot(x_range, run_data, label=label)
    all_data = np.vstack(all_data)
    print(f'{label}: {all_data.shape}')

    # mean, std = get_mean_stderr(all_data)
    # mean, std = mean_chunk_data(mean, STEP_SIZE), mean_chunk_data(std, STEP_SIZE)

    all_data = mean_chunk_data(all_data, STEP_SIZE, axis=1)
    mean, std = get_mean_stderr(all_data)

    
    x_range = get_x_range(0, mean.shape[0], STEP_SIZE)

    # label = f"OCI_{group[0]['OCI_update_interval']}_bonus_{group[0]['use_exploration_bonus']}_polyak_{group[0]['polyak_stepsize']} {postfix}"

    plot_mean_ribbon(ax, mean, std, x_range, label=label, color=color, linestyle=linestyle)

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

def create_individual_plot(ax, file_name):
    # plot_max_reward_rate(ax, file_name)
    plot_reward_rate_group(ax, file_name, None)      


if __name__ == "__main__":
    # create_individual_plot('experiments/chunlok/mpi/extended/collective/dyna_gpi_only_low_init_sweep.py', 'okay')

    # parameter_path = 'experiments/chunlok/mpi/extended_half/collective/dyna_ourgpi_maxaction.py'
    fig, ax = plt.subplots(figsize=(5, 5 / 3 * 2), dpi=300)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.rcParams.update({'font.size': 10})
    plt.tight_layout()

    # Setting styles

    #######################
    ######## PINBALL SIMPLE
    #######################
    # param_list = get_configuration_list_from_file_path('experiments/pinball/refactor/beta_gsp_learn_scratch_model_30.py')
    # complete_param_list = get_complete_configuration_list(param_list)
    # groups = group_configs(complete_param_list, ignore_keys=['seed'])

    # filtered_groups = []
    # perfs = []
    # for group in groups:

    #     beta = group[0]['oci_beta']
    #     step_size = group[0]['step_size']
    #     polyak = group[0]['polyak_stepsize']

    #     def plot_me(beta, step_size):
    #         if beta == 1.0 or beta == 0.5 or beta == 0.1 or beta == 0.05 or beta == 0.01 or beta==0.0: return True
    #         return False 

    #     if not plot_me(beta, step_size): continue
    #     plot_reward_rate_group(ax, group[1], label=f"oci_beta: {'{:0.3e}'.format(beta)}")
    
    
    #######################
    ###### PINBALL HARD
    #######################

    # color_map = {
    #     'baseline': '#34495e',
    #     'beta_1': '#3498db',
    #     'beta_5e-1': '#2ecc71',
    #     'beta_1e-1': '#f39c12',
    #     'beta_1e-3': '#e74c3c',
    # }

    # param_list = get_configuration_list_from_file_path('experiments/pinball_hard/sweep/baseline_30.py')
    # complete_param_list = get_complete_configuration_list(param_list)
    # groups = group_configs(complete_param_list, ignore_keys=['seed'])

    # for group in groups:
    #     beta = group[0]['oci_beta']
    #     step_size = group[0]['step_size']
    #     polyak = group[0]['polyak_stepsize']

    #     def plot_me(beta, step_size):
    #         if step_size == 0.0005 and polyak == 0.025: return True
    #         # return False
    #         return False
    #     if not plot_me(beta, step_size): continue
    #     plot_reward_rate_group(ax, group[1], label=f"oci_beta: {beta}", color=color_map['baseline'])
    #     # plot_reward_rate_group(ax, group[1], label=f"oci_beta: {beta} step_size: {step_size} polyak: {polyak}")

    
    # param_list = get_configuration_list_from_file_path('experiments/pinball_hard/sweep/gsp_learn_beta_ddqn_value_model_refactor_30.py')
    # # param_list = get_configuration_list_from_file_path('experiments/pinball/refactor/beta_gsp_learn_long_30.py')
    
    # # # param_list = get_configuration_list_from_file_path('experiments/pinball/refactor/beta_gsp_learn_scratch_model_30.py')
    # complete_param_list = get_complete_configuration_list(param_list)
    # groups = group_configs(complete_param_list, ignore_keys=['seed'])

    # filtered_groups = []
    # perfs = []
    # for group in groups:

    #     beta = group[0]['oci_beta']
    #     step_size = group[0]['step_size']
    #     polyak = group[0]['polyak_stepsize']

    #     def plot_me(beta, step_size):
    #         # if beta == 1.0 or beta == 0.5 or beta == 0.1 or beta == 0.05 or beta == 0.01 or beta==0.0: return True
    #         return True
    #         # return False

    #     if not plot_me(beta, step_size): continue
    #     # print(scale_value(polyak, 0, 0.1))

    #     if beta == 1: color = color_map['beta_1']
    #     if beta == 5e-1: color = color_map['beta_5e-1']
    #     if beta == 1e-1: color = color_map['beta_1e-1']
    #     if beta == 1e-3: color = color_map['beta_1e-3']
    #     plot_reward_rate_group(ax, group[1], label=f"oci_beta: {'{:0.3e}'.format(beta)}", color=color)
    
    # # Pinball hard styles

    # ax.set_xlim([0, 3000 - STEP_SIZE])
    # ax.set_ylim(-5, 55)

    # ax.set_yticks(list(range(0, 51, 10)) + [55])
    # ax.set_yticklabels(list(range(0, 51, 10)) + [55])

    # # get all the labels of this axis
    # labels = ax.get_yticklabels()
    # # remove the first and the last labels
    # labels[-1] = ""
    # # set these new labels
    # ax.set_yticklabels(labels)

    # ax.set_xticks(list(range(0, 3000, 500)) + [3000 - STEP_SIZE])
    # ax.set_xticklabels(list(range(0, 3000, 500)) + [''])
    
    ####################################
    ##### Exploration Experiment
    ######################################

    param_list = get_configuration_list_from_file_path('experiments/pinball/explore/baseline_30.py')

    complete_param_list = get_complete_configuration_list(param_list)
    groups = group_configs(complete_param_list, ignore_keys=['seed'])

    filtered_groups = []
    perfs = []
    for group in groups:

        beta = group[0]['oci_beta']
        step_size = group[0]['step_size']
        polyak = group[0]['polyak_stepsize']

        def plot_me(beta, step_size):
            # if beta == 0.1 or beta == 0.05 or beta == 0.01: return True
            if step_size == 0.001 and polyak == 0.05: return True
            return False
            # return True
            # return True

        if not plot_me(beta, step_size): continue
        # color=list(TOL_BRIGHT.values())[i]
        # print(scale_value(polyak, 0, 0.1))
        plot_reward_rate_group(ax, group[1], label=f"oci_beta: {'{:0.3e}'.format(beta)}", color = '#34495e')
        # i += 1

    param_list = get_configuration_list_from_file_path('experiments/pinball/explore/gsp_learn_beta_switch.py')

    complete_param_list = get_complete_configuration_list(param_list)
    groups = group_configs(complete_param_list, ignore_keys=['seed'])

    filtered_groups = []
    perfs = []
    for group in groups:

        beta = group[0]['oci_beta']
        step_size = group[0]['step_size']
        polyak = group[0]['polyak_stepsize']

        def plot_me(beta, step_size):
            # if beta == 0.1 or beta == 0.05 or beta == 0.01: return True
            # if step_size == 0.001 and polyak == 0.05: return True
            # return False
            return True
            # return True

        if not plot_me(beta, step_size): continue
        # color=list(TOL_BRIGHT.values())[i]
        # print(scale_value(polyak, 0, 0.1))
        plot_reward_rate_group(ax, group[1], label=f"oci_beta: {'{:0.3e}'.format(beta)}", color = '#f39c12')
        # i += 1

    ax.set_xlim([0, 1000 - STEP_SIZE])
    ax.set_ylim(-5, 200)

    ax.set_yticks(list(range(0, 201, 50)))
    # ax.set_yticklabels(list(range(0, 51, 10)) + [55])

    # # get all the labels of this axis
    # labels = ax.get_yticklabels()
    # # remove the first and the last labels
    # labels[-1] = ""
    # # set these new labels
    # ax.set_yticklabels(labels)

    ax.set_xticks(list(range(0, 1000, 200)) + [1000 - STEP_SIZE])
    ax.set_xticklabels(list(range(0, 1000, 200)) + [''])




    # Getting file name
    save_file = get_file_name('./plots/', f'reward_rate_pretty', 'pdf', timestamp=True)
    # print(f'Plot will be saved in {save_file}')
    plt.savefig(f'{save_file}')
    
    # plt.show()

        