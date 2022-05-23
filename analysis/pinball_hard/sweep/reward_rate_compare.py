
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

STEP_SIZE = 100

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
    fig, ax = plt.subplots(figsize=(10, 6))

    # # ax.set_ylim([0, 10000])
    # ax.set_xlabel('number of steps x100')
    # ax.set_ylabel('reward rate')

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
    #     plot_reward_rate_group(ax, group[1], label=f"oci_beta: {beta}")
    #     # plot_reward_rate_group(ax, group[1], label=f"oci_beta: {beta} step_size: {step_size} polyak: {polyak}")

    # colormap = cm.get_cmap('viridis')

    # param_list = get_configuration_list_from_file_path('experiments/pinball_hard/sweep/gsp_learn_display.py')
    # complete_param_list = get_complete_configuration_list(param_list)
    # groups = group_configs(complete_param_list, ignore_keys=['seed'])

    # filtered_groups = []
    # perfs = []
    # for group in groups:

    #     beta = group[0]['oci_beta']
    #     step_size = group[0]['step_size']
    #     polyak = group[0]['polyak_stepsize']

    #     def plot_me(beta, step_size):
    #         return True

    #     if not plot_me(beta, step_size): continue
    #     # print(scale_value(polyak, 0, 0.1))
    #     plot_reward_rate_group(ax, group[1], label=f"oci_beta: {'{:0.3e}'.format(beta)}", color=colormap(scale_value(beta, 0, 0.1, lambda x : x**0.14)))

    
    
    # param_list = get_configuration_list_from_file_path('experiments/pinball_hard/sweep/gsp_learn_beta_ddqn_model.py')
    # complete_param_list = get_complete_configuration_list(param_list)
    # groups = group_configs(complete_param_list, ignore_keys=['seed'])

    # filtered_groups = []
    # perfs = []
    # for group in groups:

    #     beta = group[0]['oci_beta']
    #     step_size = group[0]['step_size']
    #     polyak = group[0]['polyak_stepsize']

    #     def plot_me(beta, step_size):
    #         # if beta == 0.5: return True
    #         # return False
    #         return True

    #     if not plot_me(beta, step_size): continue
    #     # print(scale_value(polyak, 0, 0.1))
    #     plot_reward_rate_group(ax, group[1], label=f"oci_beta: {'{:0.3e}'.format(beta)}")

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

    #         if beta == 0.0: return True
    #         return False
    #         # return True
            
            

    #     if not plot_me(beta, step_size): continue
    #     # print(scale_value(polyak, 0, 0.1))
    #     plot_reward_rate_group(ax, group[1], label=f"oci_beta: {'{:0.3e}'.format(beta)}")


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
    #     plot_reward_rate_group(ax, group[1], label=f"oci_beta: {'{:0.3e}'.format(beta)}")
    
    # param_list = get_configuration_list_from_file_path('experiments/pinball/explore/gsp_learn_beta.py')

    # groups = group_configs(param_list, ignore_keys=['seed'])

    # filtered_groups = []
    # perfs = []
    # for group in groups:
    #     beta = group[0]['oci_beta']
    #     step_size = group[0]['step_size']
    #     polyak = group[0]['polyak_stepsize']

    #     def plot_me(beta, step_size):
    #         # if beta == 1.0 or beta == 0.5 or beta == 0.1 or beta == 0.05 or beta == 0.01: return True
    #         # if beta == 0.0: return True
    #         return True
    #         # return False

    #     if not plot_me(beta, step_size): continue

    #     plot_reward_rate_group(ax, group[1], label=f"oci_beta: {'{:0.3e}'.format(beta)}")

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
    #         # if beta == 1.0 or beta == 0.5 or beta == 0.1 or beta == 0.05 or beta == 0.01 or beta==0.0: return True
    #         if beta == 0.0: return True
    #         return False 

    #     if not plot_me(beta, step_size): continue
    #     plot_reward_rate_group(ax, group[1], label=f"oci_beta: {'{:0.3e}'.format(beta)}")

    # param_list = get_configuration_list_from_file_path('experiments/pinball/scratch/scratch_gsp_learn_model_ddqn_sweep.py')
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

    #     if not plot_me(beta, step_size): continue
    #     plot_reward_rate_group(ax, group[1], label=f"scratch oci_beta: {'{:0.3e}'.format(beta)}")
    
    

    param_list = get_configuration_list_from_file_path('experiments/pinball/scratch/short_scratch_model_gsp_learn_50k.py')
    complete_param_list = get_complete_configuration_list(param_list)
    groups = group_configs(complete_param_list, ignore_keys=['seed'])

    filtered_groups = []
    perfs = []
    for group in groups:

        beta = group[0]['oci_beta']
        step_size = group[0]['step_size']
        polyak = group[0]['polyak_stepsize']

        def plot_me(beta, step_size):
            # if beta == 1.0 or beta == 0.5 or beta == 0.1 or beta == 0.05 or beta == 0.01 or beta==0.0: return True
            return True 

        if not plot_me(beta, step_size): continue
        plot_reward_rate_group(ax, group[1], label=f"scratch oci_beta: {'{:0.3e}'.format(beta)}")
    



    # param_list = get_configuration_list_from_file_path('experiments/pinball/refactor/beta_gsp_learn_long_sweep.py')
    # complete_param_list = get_complete_configuration_list(param_list)
    # groups = group_configs(complete_param_list, ignore_keys=['seed'])

    # filtered_groups = []
    # perfs = []
    # for group in groups:

    #     beta = group[0]['oci_beta']
    #     step_size = group[0]['step_size']
    #     polyak = group[0]['polyak_stepsize']

    #     def plot_me(beta, step_size):
    #         if beta == 1.0 or beta == 0.5 or beta == 0.1 or beta == 0.05 or beta == 0.001: return True
    #         return False
    #         return True

    #     if not plot_me(beta, step_size): continue
    #     # print(scale_value(polyak, 0, 0.1))
    #     plot_reward_rate_group(ax, group[1], label=f"old oci_beta: {'{:0.3e}'.format(beta)}")

        


    plt.legend()
    # plt.title(alg_name)

    ax.set_xlim([-100, 3000])
    ax.set_ylim(-10, 160)

    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

    matplotlib.rc('font', **font)
    plt.yticks(fontsize=20)

    # Getting file name
    save_file = get_file_name('./plots/', f'reward_rate', 'png', timestamp=True)
    # print(f'Plot will be saved in {save_file}')
    plt.savefig(f'{save_file}')
    
    # plt.show()

        