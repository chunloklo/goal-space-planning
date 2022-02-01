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
from analysis.common import get_best_grouped_param, load_reward_rate, load_max_reward_rate, plot_mean_ribbon
from experiment_utils.sweep_configs.common import get_configuration_list_from_file_path
from  experiment_utils.analysis_common.process_line import get_mean_std, mean_chunk_data, get_mean_stderr
from pathlib import Path

STEP_SIZE = 50

def plot_best_reward_rate_individual(ax, param_file_name: str, label: str):
    parameter_list = get_configuration_list_from_file_path(param_file_name)

    # grouping configs based on seeds
    grouped_params = group_configs(parameter_list, ['seed'])
    best_group, best_index, best_performance, perfs, rank = get_best_grouped_param(grouped_params)

    print(best_group[0])
    # print(rank)
    # print([group[0]['kappa'] for group in np.array(grouped_params)[rank]])
    # print([group[0]['kappa'] for group in grouped_params])
    # adasd
    # best_group = grouped_params[rank[-3]]
    # print(grouped_params[rank[1]])
    data = load_configuration_list_data(best_group[1], load_reward_rate)

    for run_data in data:
        # run_data = window_smoothing(run_data, STEP_SIZE)
        run_data = mean_chunk_data(run_data, STEP_SIZE, 0)
        x_range = get_x_range(0, run_data.shape[0], STEP_SIZE)

        ax.plot(x_range, run_data, linewidth=0.5)

    
    # data = np.array(load_configuration_list_data(best_group[1], load_reward_rate))
    # mean, std = get_mean_stderr(data)
    # Smoothing both data
    # mean, std = mean_chunk_data(mean, STEP_SIZE, 0), mean_chunk_data(std, STEP_SIZE, 0)

    # x_range = get_x_range(0, mean.shape[0], STEP_SIZE)
    # ax.plot(x_range, mean)
    # plot_mean_ribbon(ax, mean, std, x_range, label=label)
    
def plot_max_reward_rate(ax, param_file_name: str):
    parameter_list = get_configuration_list_from_file_path(param_file_name)
    
    # doesn't matter what we do here.
    max_reward_rate = load_max_reward_rate(parameter_list[0])
    max_reward_rate = mean_chunk_data(max_reward_rate, STEP_SIZE, 0)
    x_range = get_x_range(0, max_reward_rate.shape[0], STEP_SIZE)

    # Woops, the logged max reward rate is actually incorrect. Here's the actual correct one
    # fixed_max_reward_rate = [-0.0888888889] * 799 + [-0.05384615384] * 799
    ax.plot(x_range, max_reward_rate, label='max reward rate')

def create_individual_plot(file_name, alg_name=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    if alg_name is None:
        alg_name = Path(file_name).stem

    subfolder = 'collective'

    plot_max_reward_rate(ax, file_name)
    plot_best_reward_rate_individual(ax, file_name, None)    
    # plot_max_reward_rate(ax, f'experiments/chunlok/mpi/switch_experiment/{subfolder}/dyna_sweep.py',)
    # plot_best_reward_rate(ax, f'experiments/chunlok/mpi/switch_experiment/{subfolder}/dyna_sweep.py', 'dyna')    
    plt.legend()
    plt.title(alg_name)

    # ax.set_xlim([600, 1200])

    # Getting file name
    save_file = get_file_name('./plots/', f'individual_reward_rate_{alg_name}', 'png', timestamp=True)
    # print(f'Plot will be saved in {save_file}')
    plt.savefig(f'{save_file}', dpi = 300)


if __name__ == "__main__":
    subfolder = 'collective'

    # create_individual_plot(f'experiments/chunlok/mpi/extended/collective/dyna_gpi_only_low_init_sweep.py', 'gpi')
    # create_individual_plot(f'experiments/chunlok/mpi/switch_experiment/{subfolder}/dyna_backgroundgpi_sweep.py', f'{subfolder} backgroundGPI')
    # create_individual_plot(f'experiments/chunlok/mpi/switch_experiment/{subfolder}/dyna_gpi_sweep.py', f'{subfolder} GPI')
    # create_individual_plot(f'experiments/chunlok/mpi/switch_experiment/{subfolder}/dynaoptions_sweep.py', f'{subfolder} dynaoptions')

    # create_individual_plot(f'experiments/chunlok/mpi/switch_experiment/{subfolder}/dyna_backgroundgpi_only_sweep.py', f'{subfolder} low init OurGPI')

    # create_individual_plot('experiments/chunlok/mpi/extended/collective/dyna_backgroundgpi_only_low_init_sweep.py', 'dyna')
    # plt.show()

    

    create_individual_plot('experiments/chunlok/graze_learn/OCI.py')

    # folder = 'experiments/chunlok/mpi/extended_half/collective/'
    # files = [
    #     # {
    #     #     'file' : 'dyna_backgroundgpi_only_low_init_sweep.py',
    #     #     'label' : 'ourGPI'
    #     # },
    #     {
    #         'file' : 'dyna_ourgpi_maxaction.py',
    #         'label' : 'maxAction OurGPI'
    #     },
    #     # {
    #     #     'file' : 'dyna_gpi_only_low_init_sweep.py',
    #     #     'label' : 'GPI'
    #     # },
    #     # {
    #     #     'file' : 'dyna_sweep.py',
    #     #     'label' : 'dyna'
    #     # },
    #     # {
    #     #     'file' : 'dynaoptions_sweep.py',
    #     #     'label' : 'dynaoptions'
    #     # }
        
    # ]

    # for obj in files:
    #     create_individual_plot(folder + obj['file'], obj['label'])
    

        