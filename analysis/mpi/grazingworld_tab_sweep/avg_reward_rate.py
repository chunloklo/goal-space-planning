import os
import sys
sys.path.append(os.getcwd())

from typing import Any, Callable, Dict, List, Tuple
import argparse
from src.utils import run_utils
import importlib.util
from copy import copy
from sweep_configs.analysis import group_configs
import numpy as np
from pprint import pprint
import matplotlib
from src.analysis.gridworld_utils import _get_corner_loc, _get_q_value_patch_paths, get_text_location, prompt_user_for_file_name, get_file_name, scale_value, _plot_init, prompt_episode_display_range
from src.analysis.plot_utils import load_experiment_data, get_json_handle, get_x_range
import matplotlib.pyplot as plt
from sweep_configs.common import get_configuration_list_from_file_path

from src.analysis.plot_utils import load_parameter_list_data, get_mean_std, mean_chunk_data

# You can likely cache this for better performance later.
def load_reward_rate(param):
    data = run_utils.load_data(param)
    # 1 x num_logs
    data = np.array(data['reward_rate'])
    return data[0]

def load_max_reward_rate(param):
    data = run_utils.load_data(param)
    # 1 x num_logs
    data = np.array(data['max_reward_rate'])
    return data[0]
    
def get_perf(parameter_list):
    data = np.array(load_parameter_list_data(parameter_list, load_reward_rate))
    return np.mean(data)


STEP_SIZE = 20

def plot_best_reward_rate(ax, param_file_name: str, label: str):
    parameter_list = get_configuration_list_from_file_path(param_file_name)

    # grouping configs based on seeds
    grouped_params = group_configs(parameter_list, ['seed'])
    perfs = [get_perf(param_list) for _, param_list in grouped_params]
    best_index = np.argmax(perfs)
    print(perfs)
    # best_index = 5

    pprint(grouped_params[best_index][0])

    data = np.array(load_parameter_list_data(grouped_params[best_index][1], load_reward_rate))
    mean, std = get_mean_std(data)
    mean, std = mean_chunk_data(mean, STEP_SIZE, 0), mean_chunk_data(std, STEP_SIZE, 0)

    x_range = get_x_range(0, mean.shape[0], STEP_SIZE)

    line = ax.plot(x_range, mean, label=label)
    ax.fill_between(x_range, mean + std, mean - std, alpha=0.25, color=line[0].get_color())

def plot_max_reward_rate(ax, param_file_name: str):
    parameter_list = get_configuration_list_from_file_path(param_file_name)
    
    # doesn't matter what we do here.
    max_reward_rate = load_max_reward_rate(parameter_list[0])
    max_reward_rate = mean_chunk_data(max_reward_rate, STEP_SIZE, 0)
    x_range = get_x_range(0, max_reward_rate.shape[0], STEP_SIZE)

    fixed_max_reward_rate = [-0.05384615384] * len(list(x_range))

    # ax.plot(x_range, max_reward_rate, label='max reward rate')
    # ax.plot(x_range, fixed_max_reward_rate, label='fixed max reward rate')


    fixed_max_reward_rate = [-0.05384615384] * len(list(x_range))
    fixed_lower_max_reward_rate = [-0.0888888889]* len(list(x_range))

    fixed_max_reward_rate = [-0.0888888889] * 799 + [-0.05384615384] * 799
    # ax.plot(x_range, max_reward_rate, label='max reward rate')
    # ax.plot(x_range, fixed_max_reward_rate, label='fixed max reward rate')
    ax.plot(fixed_max_reward_rate, label='max reward rate')


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(10, 6))

    subfolder = 'collective'

    plot_max_reward_rate(ax, f'experiments/chunlok/mpi/switch_experiment/{subfolder}/dyna_sweep.py',)
    plot_best_reward_rate(ax, f'experiments/chunlok/mpi/switch_experiment/{subfolder}/dyna_sweep.py', 'dyna')
    # plot_best_reward_rate(ax, f'experiments/chunlok/mpi/switch_experiment/{subfolder}/dyna_backgroundgpi_only_sweep.py', 'OurGPI')
    # plot_best_reward_rate(ax, f'experiments/chunlok/mpi/switch_experiment/individual/dyna_backgroundgpi_only_sweep.py', 'iOurGPI')
    # plot_best_reward_rate(ax, f'experiments/chunlok/mpi/switch_experiment/{subfolder}/dyna_gpi_only_sweep.py', 'gpi') 
    plot_best_reward_rate(ax, f'experiments/chunlok/mpi/switch_experiment/{subfolder}/dyna_backgroundgpi_only_low_init_sweep.py', 'OurGPI low init')
    plot_best_reward_rate(ax, f'experiments/chunlok/mpi/switch_experiment/individual/dyna_backgroundgpi_only_low_init_sweep.py', 'iOurGPI low init')
    plot_best_reward_rate(ax, f'experiments/chunlok/mpi/switch_experiment/{subfolder}/dyna_gpi_only_low_init_sweep.py', 'gpi low init') 
    # plot_best_reward_rate(ax, f'experiments/chunlok/mpi/switch_experiment/{subfolder}/dyna_backgroundgpi_sweep.py', 'dyna_background_gpi')
    # plot_best_reward_rate(ax, f'experiments/chunlok/mpi/switch_experiment/{subfolder}/dyna_gpi_sweep.py', 'dyna_gpi') 

    plot_best_reward_rate(ax, f'experiments/chunlok/mpi/switch_experiment/{subfolder}/dynaoptions_sweep.py', 'dynaoptions')       
    plt.legend()
    plt.title(subfolder)

    # ax.set_xlim([600, 1200])

    # Getting file name
    save_file = get_file_name('./plots/', 'reward_rate', 'png', timestamp=True)
    print(f'Plot will be saved in {save_file}')

    plt.savefig(f'{save_file}', dpi = 300)

    # plt.show()

        