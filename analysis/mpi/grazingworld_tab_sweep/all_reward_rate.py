
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
from tqdm import tqdm

STEP_SIZE = 20

def get_best_config_from_file_path_cached(config_file_name, get_cached=False):
    @cache_local_file('local_cache.pkl', config_file_name, get_cached)
    def cache_func():
        parameter_list = get_configuration_list_from_file_path(config_file_name)
        print(len(parameter_list))
        # grouping configs based on seeds
        grouped_params = group_configs(parameter_list, ['seed'])
        result = get_best_grouped_param(grouped_params)
        return result
        
    result = cache_func()
    return result

def plot_all_reward_rate(ax, param_file_name: str, label: str):
    parameter_list = get_configuration_list_from_file_path(param_file_name)
    # print("plotting only the first index of the config list")

    ##### KAPPA SWEEP
    grouped_params = group_configs(parameter_list, ['seed'])

    for group in tqdm(grouped_params):
        # if group[0]['alpha'] == 0.9:
        kappa = group[0]['skip_threshold']

        data = np.array(load_configuration_list_data(group[1], load_reward_rate))
        mean, std = get_mean_stderr(data)
        # Smoothing both data
        mean, std = mean_chunk_data(mean, STEP_SIZE, 0), mean_chunk_data(std, STEP_SIZE, 0)

        x_range = get_x_range(0, mean.shape[0], STEP_SIZE*group[1][0]['step_logging_interval'])

        plot_mean_ribbon(ax, mean, std, x_range, label=str(kappa))

    ##### ####

    # GETTING SPECIFIC INDEX
    # grouped_params = group_configs(parameter_list, ['seed'])
    # for group in grouped_params:
    #     if group[0]['alpha'] == 1.0 and np.isclose(group[0]['kappa'],0.03):
    #         data = load_configuration_list_data(group[1], load_reward_rate)
    #         for run_data in data:
    #             run_data = mean_chunk_data(run_data, STEP_SIZE, 0)
    #             x_range = get_x_range(0, run_data.shape[0], STEP_SIZE*group[1][0]['step_logging_interval'])

    #             ax.plot(x_range, run_data, linewidth=0.5)



    

    # ######### GETTING BEST INDEX
    # best_group, best_index, best_performance, perfs, rank = get_best_config_from_file_path_cached(param_file_name, True)
    # print(best_group[0])
    # data = load_configuration_list_data(best_group[1], load_reward_rate)

    # for run_data in data:
    #     # run_data = window_smoothing(run_data, STEP_SIZE)
    #     run_data = mean_chunk_data(run_data, STEP_SIZE, 0)
    #     x_range = get_x_range(0, run_data.shape[0], STEP_SIZE*best_group[1][0]['step_logging_interval'])

    #     ax.plot(x_range, run_data, linewidth=0.5)
    
    
def plot_max_reward_rate(ax, param_file_name: str):
    parameter_list = get_configuration_list_from_file_path(param_file_name)
    parameter_list = [parameter_list[0]]
    
    # doesn't matter what we do here.
    max_reward_rate = load_data(parameter_list[0], 'max_reward_rate')
    # max_reward_rate = max_reward_rate[344:]
    print(max_reward_rate.shape)
    max_reward_rate = mean_chunk_data(max_reward_rate, STEP_SIZE, 0)
    x_range = get_x_range(0, max_reward_rate.shape[0], STEP_SIZE*parameter_list[0]['step_logging_interval'])
    

    ax.plot(x_range, max_reward_rate, label='max reward rate')

    # fixed_max_reward_rate = [-0.05384615384] * len(list(x_range))
    # fixed_lower_max_reward_rate = [-0.0888888889]* len(list(x_range))

    # fixed_max_reward_rate = [-0.0888888889] * 799 + [-0.05384615384] * 799
    # ax.plot(x_range, max_reward_rate, label='max reward rate')
    # ax.plot(x_range, fixed_max_reward_rate, label='fixed max reward rate')
    # ax.plot(fixed_max_reward_rate, label='max reward rate')

def create_individual_plot(file_name, alg_name = None):
    if alg_name is None:
        alg_name = Path(file_name).stem

    fig, ax = plt.subplots(figsize=(10, 6))

    plot_max_reward_rate(ax, file_name)
    plot_all_reward_rate(ax, file_name, None)    
    # plot_max_reward_rate(ax, f'experiments/chunlok/mpi/switch_experiment/{subfolder}/dyna_sweep.py',)
    # plot_best_reward_rate(ax, f'experiments/chunlok/mpi/switch_experiment/{subfolder}/dyna_sweep.py', 'dyna')    
    plt.legend()
    plt.title(alg_name)

    # ax.set_xlim([600, 1200])

    # Getting file name
    save_file = get_file_name('./plots/', f'individual_reward_rate_{alg_name}', 'png', timestamp=True)
    print(f'Plot will be saved in {save_file}')
    plt.savefig(f'{save_file}', dpi = 300)


if __name__ == "__main__":

    # create_individual_plot('experiments/chunlok/mpi/extended/collective/dyna_gpi_only_low_init_sweep.py', 'okay')

    # parameter_path = 'experiments/chunlok/mpi/extended_half/collective/dyna_ourgpi_maxaction.py'
    # create_individual_plot('experiments/chunlok/graze_ideal/dyna.py')
    # create_individual_plot('experiments/chunlok/graze_ideal/dynaoptions.py')
    # create_individual_plot('experiments/chunlok/graze_ideal/OCG.py')
    # create_individual_plot('experiments/chunlok/graze_ideal/OCI_action.py')
    # create_individual_plot('experiments/chunlok/graze_ideal/OCI.py')

    # create_individual_plot('experiments/chunlok/graze_ideal/OCG_goal.py')
    # create_individual_plot('experiments/chunlok/graze_ideal/OCI_goal.py')
    create_individual_plot('experiments/chunlok/env_tmaze/baseline.py')
    # plt.show()

        