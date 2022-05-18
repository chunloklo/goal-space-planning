
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
import matplotlib.colors as mcolors

STEP_SIZE = 20

# Plots the reward rate for a single run. Mainly used for debugging

def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]


def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

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

        run_data = mean_chunk_data(data, STEP_SIZE, 0)
        # print(run_data.shape)

        x_range = get_x_range(0, run_data.shape[0], STEP_SIZE)

        # print(len(list(x_range)))
        ax.plot(x_range, run_data, color=color, alpha=0.05)

    all_data = np.vstack(all_data)
    print(all_data.shape)

    mean, std = get_mean_stderr(all_data)

    mean, std = mean_chunk_data(mean, STEP_SIZE), mean_chunk_data(std, STEP_SIZE)
    x_range = get_x_range(0, mean.shape[0], STEP_SIZE)

    ax.plot(x_range, mean, label=param_file_name, color=color, alpha=1.0)



def plot_reward_rate_group(ax, group, label, color=None):
    all_data = []
    line = None

    # if color == '#f39c12':
    #     cmap = get_continuous_cmap(['#d35400', '##f1c40f'])
    # else:
    #     cmap = get_continuous_cmap(['#2c3e50', '#ecf0f1'])

    cmap = get_continuous_cmap([color, color])

    cmap_val = np.linspace(0, 1, len(group))

    perfs = []

    for i, param in enumerate(group):
        ############ STANDARD
        data = load_data(param, 'reward_rate')
        # print(f'label: {label} seed: {param["seed"]}, {np.mean(data)}')
        all_data.append(data)

        perfs.append(np.mean(data))

    ind = np.argsort(perfs)

    for i, index in enumerate(ind):

        data = all_data[index]
        # print(i, np.mean(data))
        run_data = mean_chunk_data(data, STEP_SIZE, 0)
        x_range = get_x_range(0, run_data.shape[0], STEP_SIZE)

        plot_color = cmap(cmap_val[i])

        # # print(len(list(x_range)))
        # if line is None:
        #     plot_color = color
        # else:
        #     plot_color = line[-1].get_color()
        if i == round(0.0 * (len(ind) - 1)) or i == round(1.0 * (len(ind) - 1)):
            alpha = 0.5
        else:
            alpha = 0.05
        line = ax.plot(x_range, run_data, color=plot_color, alpha=alpha)

    all_data = np.vstack(all_data)
    print(all_data.shape)

    mean, std = get_mean_stderr(all_data)

    mean, std = mean_chunk_data(mean, STEP_SIZE), mean_chunk_data(std, STEP_SIZE)
    x_range = get_x_range(0, mean.shape[0], STEP_SIZE)

    # plot_color = line[-1].get_color()
    plot_color = color
    ax.plot(x_range, mean, label=label, color=plot_color, alpha=1.0, linestyle='dashed', linewidth=2)

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
    fig, ax = plt.subplots(figsize=(5, 5 / 3 * 2), dpi=600)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.rcParams.update({'font.size': 10})
    plt.tight_layout()

    # ax.set_ylim([0, 10000])
    # ax.set_xlabel('steps x100')
    # ax.set_ylabel('reward rate')

    i = 0

    # param_list = get_configuration_list_from_file_path('experiments/pinball_hard/sweep/baseline.py')
    # complete_param_list = get_complete_configuration_list(param_list)
    # groups = group_configs(complete_param_list, ignore_keys=['seed'])

    # filtered_groups = []
    # perfs = []
    # for group in groups:
    #     beta = group[0]['oci_beta']
    #     step_size = group[0]['step_size']
    #     polyak = group[0]['polyak_stepsize']

    #     def plot_me(beta, step_size):
    #         # if step_size == 1e-4: return True
    #         # if beta == 0.1 or beta == 0.0: return True
    #         # if beta == 0.1: return True
    #         if polyak == 0.025: return True
    #         # return True
    #         return False

    #     if not plot_me(beta, step_size): continue

    #     color=list(TOL_BRIGHT.values())[i]
    #     # color=None

    #     plot_reward_rate_group(ax, group[1], label=f"oci_beta: {beta} step_size: {step_size} polyak: {polyak}", color=color)
    #     i += 1

    # param_list = get_configuration_list_from_file_path('experiments/pinball_hard/sweep/gsp_learn_more.py')

    # complete_param_list = get_complete_configuration_list(param_list)

    # groups = group_configs(complete_param_list, ignore_keys=['seed'])
    # i = 0

    # filtered_groups = []
    # perfs = []
    # for group in groups:
    #     # if group[0]['use_exploration_bonus'] == False:
    #     #     plot_reward_rate_group(ax, group[1])

    #     # if group[0]['OCI_update_interval'] == 2 and  group[0]['use_exploration_bonus'] == True and group[0]['polyak_stepsize'] == 0.05:
    #     #     plot_reward_rate_group(ax, group[1])
    #     # if group[0]['oci_beta'] == 0.25:
    #     # print(group[0]['oci_beta'])

    #     beta = group[0]['oci_beta']
    #     step_size = group[0]['step_size']
    #     polyak = group[0]['polyak_stepsize']

    #     def plot_me(beta, step_size):
    #         # if step_size == 1e-4: return True
    #         # if beta == 0.1 or beta == 0.0: return True
    #         # if beta == 0.1: return True
    #         if beta == 0.2: return True
    #         # return True
    #         # return False

    #     if not plot_me(beta, step_size): continue

    #     color=list(TOL_BRIGHT.values())[i]
    #     # color=None

    #     plot_reward_rate_group(ax, group[1], label=f"oci_beta: {beta} step_size: {step_size} polyak: {polyak}", color=color)
    #     i += 1


    # param_list = get_configuration_list_from_file_path('experiments/pinball_hard/sweep/gsp_learn_single.py')
    # complete_param_list = get_complete_configuration_list(param_list)
    # groups = group_configs(complete_param_list, ignore_keys=['seed'])
    # for group in groups:
    #     beta = group[0]['oci_beta']
    #     step_size = group[0]['step_size']
    #     polyak = group[0]['polyak_stepsize']

    #     def plot_me(beta, step_size, polyak):
    #         # if step_size == 1e-4: return True
    #         # if beta == 0.1 or beta == 0.0: return True
    #         # if beta == 0.1: return True
    #         # if step_size == 5e-4 and polyak == 0.025: return True
    #         # if beta < .0125: return True
    #         return True
    #         # return False

    #     if not plot_me(beta, step_size, polyak): continue

    #     color=list(TOL_BRIGHT.values())[i]
    #     # color=None

    #     plot_reward_rate_group(ax, group[1], label=f"oci_beta: {beta} step_size: {step_size} polyak: {polyak}", color=color)
    #     i += 1


    # param_list = get_configuration_list_from_file_path('experiments/pinball_hard/sweep/baseline.py')
    # complete_param_list = get_complete_configuration_list(param_list)
    # groups = group_configs(complete_param_list, ignore_keys=['seed'])

    # filtered_groups = []
    # perfs = []
    # for group in groups:
    #     beta = group[0]['oci_beta']
    #     step_size = group[0]['step_size']
    #     polyak = group[0]['polyak_stepsize']

    #     def plot_me(beta, step_size):
    #         if step_size == 0.0005 and polyak == 0.025: return True
    #         # return False
    #         return False
    #     if not plot_me(beta, step_size): continue
    #     color=list(TOL_BRIGHT.values())[i]
    #     plot_reward_rate_group(ax, group[1], label=f"oci_beta: {beta}", color=color)
    #     # plot_reward_rate_group(ax, group[1], label=f"oci_beta: {beta} step_size: {step_size} polyak: {polyak}")
    #     i += 1
    

    
    # param_list = get_configuration_list_from_file_path('experiments/pinball_hard/refactor/gsp_learn_ddqn_value_policy_model.py')
    # complete_param_list = get_complete_configuration_list(param_list)
    # groups = group_configs(complete_param_list, ignore_keys=['seed'])

    # filtered_groups = []
    # perfs = []
    # for group in groups:

    #     beta = group[0]['oci_beta']
    #     step_size = group[0]['step_size']
    #     polyak = group[0]['polyak_stepsize']

    #     def plot_me(beta, step_size):
    #         # if beta == 1e-7: return True
    #         return True
    #         # return True

    #     if not plot_me(beta, step_size): continue
    #     color=list(TOL_BRIGHT.values())[i]
    #     # print(scale_value(polyak, 0, 0.1))
    #     plot_reward_rate_group(ax, group[1], label=f"oci_beta: {'{:0.3e}'.format(beta)}", color = color)
    #     i += 1


    # param_list = get_configuration_list_from_file_path('experiments/pinball_hard/sweep/gsp_learn_beta_ddqn_value_model.py')
    # param_list = get_configuration_list_from_file_path('experiments/pinball/refactor/beta_gsp_learn_long_sweep.py')
    # param_list = get_configuration_list_from_file_path('experiments/pinball/refactor/beta_gsp_learn_scratch_model.py')
    # param_list = get_configuration_list_from_file_path('experiments/pinball_hard/sweep/baseline_30.py')

    # param_list = get_configuration_list_from_file_path('experiments/pinball/explore/gsp_learn_beta.py')
    # param_list = get_configuration_list_from_file_path('experiments/pinball/explore/gsp_learn_beta_switch.py')


    #####################################
    ###### SIMPLE PINBALL
    #######################################
    
    # param_list = get_configuration_list_from_file_path('experiments/pinball/refactor/beta_gsp_learn_scratch_model_30.py')
    # # param_list = get_configuration_list_from_file_path('experiments/pinball/refactor/beta_gsp_learn_scratch_model_display.py')
    # complete_param_list = get_complete_configuration_list(param_list)
    # groups = group_configs(complete_param_list, ignore_keys=['seed'])

    # filtered_groups = []
    # perfs = []
    # for group in groups:

    #     beta = group[0]['oci_beta']
    #     step_size = group[0]['step_size']
    #     polyak = group[0]['polyak_stepsize']

    #     def plot_me(beta, step_size):
    #         # if beta == 0.1 or beta == 0.05 or beta == 0.01: return True
    #         if beta == 0.0: return True
    #         return False
    #         return True

    #     if not plot_me(beta, step_size): continue
    #     color=list(TOL_BRIGHT.values())[i]
    #     # print(scale_value(polyak, 0, 0.1))
    #     plot_reward_rate_group(ax, group[1], label=f"oci_beta: {'{:0.3e}'.format(beta)}", color = color)
    #     i += 1
    
    # param_list = get_configuration_list_from_file_path('experiments/pinball/scratch/scratch_gsp_learn_model_dqn_sweep.py')
    # complete_param_list = get_complete_configuration_list(param_list)
    # print(len(complete_param_list))
    # groups = group_configs(complete_param_list, ignore_keys=['seed'])

    # filtered_groups = []
    # perfs = []
    # for group in groups:

    #     beta = group[0]['oci_beta']
    #     step_size = group[0]['goal_learner_step_size']
    #     polyak = group[0]['goal_learner_polyak_stepsize']

    #     # use_reward = group[0]['use_reward_for_model_policy'] 
    #     # print(step_size, polyak, use_reward)


    #     def plot_me(beta, step_size):
    #         # if beta == 1.0 or beta == 0.5 or beta == 0.1 or beta == 0.05 or beta == 0.01 or beta==0.0: return True
    #         # if step_size == 1e-3: return True
    #         return True 
    #         # if step_size == 0.001 and 
    #         # if group[0]['use_reward_for_model_policy'] == True: return True
    #         # return False

    #     # color=list(TOL_BRIGHT.values())[i]
    #     if not plot_me(beta, step_size): continue
    #     plot_reward_rate_group(ax, group[1], label=f"{step_size}, {polyak}")
    #     i += 1
    



    

    ################################
    ###### SCRATCH
    #################################


    # param_list = get_configuration_list_from_file_path('experiments/pinball/scratch/scratch_gsp_learn_prefill.py')
    # complete_param_list = get_complete_configuration_list(param_list)
    # groups = group_configs(complete_param_list, ignore_keys=['seed'])

    # filtered_groups = []
    # perfs = []
    # for group in groups:

    #     beta = group[0]['oci_beta']
    #     step_size = group[0]['step_size']
    #     polyak = group[0]['polyak_stepsize']

    #     def plot_me(beta, step_size):
    #         # if beta == 0.1 or beta == 0.05 or beta == 0.01: return True
    #         # if beta == 0.0: return True
    #         # return False
    #         return True

    #     if not plot_me(beta, step_size): continue
    #     color=list(TOL_BRIGHT.values())[i]
    #     # print(scale_value(polyak, 0, 0.1))
    #     plot_reward_rate_group(ax, group[1], label=f"oci_beta: {'{:0.3e}'.format(beta)}", color = color)
    #     i += 1

    #####################################
    # Explore Env
    #####################################

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
        color=list(TOL_BRIGHT.values())[i]
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
        color=list(TOL_BRIGHT.values())[i]
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



    # param_list = get_configuration_list_from_file_path('experiments/pinball/explore/gsp_learn_beta_online_debug.py')

    # complete_param_list = get_complete_configuration_list(param_list)
    # groups = group_configs(complete_param_list, ignore_keys=['seed'])

    # filtered_groups = []
    # perfs = []
    # for group in groups:

    #     beta = group[0]['oci_beta']
    #     step_size = group[0]['step_size']
    #     polyak = group[0]['polyak_stepsize']

    #     def plot_me(beta, step_size):
    #         # if beta == 0.1 or beta == 0.05 or beta == 0.01: return True
    #         # if step_size == 0.001 and polyak == 0.05: return True
    #         # return False
    #         return True
    #         # return True

    #     if not plot_me(beta, step_size): continue
    #     color=list(TOL_BRIGHT.values())[i]
    #     # print(scale_value(polyak, 0, 0.1))
    #     plot_reward_rate_group(ax, group[1], label=f"prefill oci_beta: {'{:0.3e}'.format(beta)}", color = color)
    #     i += 1

    # param_list = get_configuration_list_from_file_path('experiments/pinball/explore/gsp_learn_beta_online_true.py')

    # complete_param_list = get_complete_configuration_list(param_list)
    # groups = group_configs(complete_param_list, ignore_keys=['seed'])

    # filtered_groups = []
    # perfs = []
    # for group in groups:

    #     beta = group[0]['oci_beta']
    #     step_size = group[0]['step_size']
    #     polyak = group[0]['polyak_stepsize']

    #     def plot_me(beta, step_size):
    #         # if beta == 0.1 or beta == 0.05 or beta == 0.01: return True
    #         # if step_size == 0.001 and polyak == 0.05: return True
    #         # return False
    #         return True
    #         # return True

    #     if not plot_me(beta, step_size): continue
    #     color=list(TOL_BRIGHT.values())[i]
    #     # print(scale_value(polyak, 0, 0.1))
    #     plot_reward_rate_group(ax, group[1], label=f"online oci_beta: {'{:0.3e}'.format(beta)}", color = color)
    #     i += 1

        


    # param_list = get_configuration_list_from_file_path('experiments/pinball/explore/gsp_learn_beta_online_debug.py')

    # complete_param_list = get_complete_configuration_list(param_list)
    # groups = group_configs(complete_param_list, ignore_keys=['seed'])

    # filtered_groups = []
    # perfs = []
    # for group in groups:

    #     beta = group[0]['oci_beta']
    #     step_size = group[0]['step_size']
    #     polyak = group[0]['polyak_stepsize']

    #     def plot_me(beta, step_size):
    #         # if beta == 0.1 or beta == 0.05 or beta == 0.01: return True
    #         # if step_size == 0.001 and polyak == 0.05: return True
    #         # return False
    #         return True
    #         # return True

    #     if not plot_me(beta, step_size): continue
    #     color=list(TOL_BRIGHT.values())[i]
    #     # print(scale_value(polyak, 0, 0.1))
    #     plot_reward_rate_group(ax, group[1], label=f"online oci_beta: {'{:0.3e}'.format(beta)}", color = color)
    #     i += 1




    

    # plt.legend()
    # ax.set_ylim([-20, 180])
    # ax.set_xlim([-10, 500])

    # Getting file name
    save_file = get_file_name('./plots/', f'reward_rate', 'pdf', timestamp=True)
    # print(f'Plot will be saved in {save_file}')
    plt.savefig(f'{save_file}', dpi = 300)
    
    # plt.show()

        