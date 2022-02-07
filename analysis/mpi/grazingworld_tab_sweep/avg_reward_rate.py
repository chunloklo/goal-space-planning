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
from  experiment_utils.analysis_common.process_line import get_mean_std, mean_chunk_data, get_mean_stderr
from  experiment_utils.analysis_common.cache import cache_local_file
from pathlib import Path

STEP_SIZE = 10

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

def plot_best_reward_rate(ax, param_file_name: str, label: str = None):
    
    if label is None:
        label = Path(param_file_name).stem

    # if label == 'dyna':
    #     parameter_list = get_configuration_list_from_file_path(param_file_name)
    #     grouped_params = group_configs(parameter_list, ['seed'])
    #     for group in grouped_params:
    #         if group[0]['alpha'] == 1.0 and np.isclose(group[0]['kappa'],0.03):
    #             data = np.array(load_configuration_list_data(group[1], load_reward_rate))
    #             # print(data.shape)
    #             # data = np.array(load_configuration_list_data(best_group[1], load_reward_rate))
    #             mean, std = get_mean_stderr(data)
    #             # Smoothing both data
    #             mean, std = mean_chunk_data(mean, STEP_SIZE, 0), mean_chunk_data(std, STEP_SIZE, 0)

    #             x_range = get_x_range(0, mean.shape[0], STEP_SIZE*group[1][0]['step_logging_interval'])

    #             plot_mean_ribbon(ax, mean, std, x_range, label=label)

    #     return

    best_group, best_index, best_performance, perfs, rank = get_best_config_from_file_path_cached(param_file_name, True)

    data = np.array(load_configuration_list_data(best_group[1], load_reward_rate))
    mean, std = get_mean_stderr(data)
    # Smoothing both data
    mean, std = mean_chunk_data(mean, STEP_SIZE, 0), mean_chunk_data(std, STEP_SIZE, 0)

    x_range = get_x_range(0, mean.shape[0], STEP_SIZE*best_group[1][0]['step_logging_interval'])

    plot_mean_ribbon(ax, mean, std, x_range, label=label)

def plot_max_reward_rate(ax, param_file_name: str):
    parameter_list = get_configuration_list_from_file_path(param_file_name)
    
    # doesn't matter what we do here.
    max_reward_rate = load_max_reward_rate(parameter_list[0])
    max_reward_rate = mean_chunk_data(max_reward_rate, STEP_SIZE, 0)
    x_range = get_x_range(0, max_reward_rate.shape[0], STEP_SIZE*parameter_list[0]['step_logging_interval'])

    # Woops, the logged max reward rate is actually incorrect. Here's the actual correct one
    # fixed_max_reward_rate = [-0.0888888889] * 799 + [-0.05384615384] * 799
    ax.plot(x_range, max_reward_rate, label='max reward rate')

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(10, 6))

    subfolder = 'collective'

    # plot_max_reward_rate(ax, f'experiments/chunlok/mpi/switch_experiment/{subfolder}/dyna_sweep.py',)
    # plot_best_reward_rate(ax, f'experiments/chunlok/mpi/switch_experiment/{subfolder}/dyna_sweep.py', 'dyna')
    # # plot_best_reward_rate(ax, f'experiments/chunlok/mpi/switch_experiment/{subfolder}/dyna_backgroundgpi_only_sweep.py', 'OurGPI')
    # # plot_best_reward_rate(ax, f'experiments/chunlok/mpi/switch_experiment/individual/dyna_backgroundgpi_only_sweep.py', 'iOurGPI')
    # # plot_best_reward_rate(ax, f'experiments/chunlok/mpi/switch_experiment/{subfolder}/dyna_gpi_only_sweep.py', 'gpi') 
    # plot_best_reward_rate(ax, f'experiments/chunlok/mpi/switch_experiment/{subfolder}/dyna_backgroundgpi_only_low_init_sweep.py', 'OurGPI low init')
    # # plot_best_reward_rate(ax, f'experiments/chunlok/mpi/switch_experiment/individual/dyna_backgroundgpi_only_low_init_sweep.py', 'iOurGPI low init')
    # plot_best_reward_rate(ax, f'experiments/chunlok/mpi/switch_experiment/{subfolder}/dyna_gpi_only_low_init_sweep.py', 'gpi low init') 
    # # plot_best_reward_rate(ax, f'experiments/chunlok/mpi/switch_experiment/{subfolder}/dyna_backgroundgpi_sweep.py', 'dyna_background_gpi')
    # # plot_best_reward_rate(ax, f'experiments/chunlok/mpi/switch_experiment/{subfolder}/dyna_gpi_sweep.py', 'dyna_gpi') 

    # plot_best_reward_rate(ax, f'experiments/chunlok/mpi/switch_experiment/{subfolder}/dynaoptions_sweep.py', 'dynaoptions')       


    folder = 'experiments/chunlok/graze_learn/'
    files = [
        {
            'file' : 'dyna.py',
        },
        {
            'file' : 'dynaoptions.py',
        },
        {
            'file' : 'OCG.py',
        },
        {
            'file' : 'OCI_action.py',
        },
        {
            'file' : 'OCI.py',
        },
        {
            'file' : 'dyna_64.py',
        }
        
    ]
    plot_max_reward_rate(ax, folder + files[0]['file'])

    for obj in files:
        plot_best_reward_rate(ax, folder + obj['file'])


    # plot_max_reward_rate(ax, 'experiments/chunlok/graze_learn/dyna_64.py')
    # plot_best_reward_rate(ax, 'experiments/chunlok/graze_learn/dyna_64.py')



    
    plt.legend()
    plt.title('Learning')

    # ax.set_xlim([600, 1200])

    # Getting file name
    save_file = get_file_name('./plots/', 'reward_rate', 'png', timestamp=True)
    print(f'Plot will be saved in {save_file}')

    plt.savefig(f'{save_file}', dpi = 300)

    # plt.show()

        