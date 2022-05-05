from typing import Any, Callable, List, Tuple
from experiment_utils.analysis_common.cache import cache_local_file
from experiment_utils.analysis_common.configs import group_configs
import numpy as np
from experiment_utils.data_io.io.zodb_io import BatchDBAccess, DB_FOLDER
from src.utils import run_utils
from experiment_utils.sweep_configs.common import get_configuration_list_from_file_path
from experiment_utils.data_io.configs import get_folder_name, load_data_from_config_zodb
from tqdm import tqdm

# Some common functions for analysis specific for this project

# You can likely cache this for better performance later.
def load_reward_rate(config):
    # data = run_utils.load_data(config)

    # # Just getting rid of the first dimension
    # data = np.array(data['reward_rate'])
    # return data[0]

    return load_data(config, 'reward_rate')

def load_max_reward_rate(config):
    # data = run_utils.load_data(config)
    # # 1 x num_logs
    # data = np.array(data['max_reward_rate'])

    # Just getting rid of the first dimension
    # return data[0]
    return load_data(config, 'max_reward_rate')

def load_data(config: dict, key: str):
    # data = run_utils.load_data(config)
    data = load_data_from_config_zodb(config)
    # 1 x num_logs
    data = np.array(data[key])

    # print(data.shape)

    # Just getting rid of the first dimension
    return data[0]

def get_performance(config):
    data = load_reward_rate(config)
    return np.mean(data)

def get_best_grouped_param(grouped_params: list[Tuple[Any, List]]) -> Tuple[Tuple[Any, List], int, float, list[float]]:
    """Return the best configuration from the group of parameters

    Args:
        configuration_list (list): List of configurations to sample from

    Returns:
        Tuple[Any, int, float, list[float]]: Best config group, best index, best performance, list of performances
    """

    db_folder = get_folder_name(grouped_params[0][0], DB_FOLDER)

    with BatchDBAccess(db_folder):
        perfs = [np.mean([get_performance(config) for config in config_list]) for _, config_list in tqdm(grouped_params)]
    rank = np.argsort(-np.array(perfs))

    
    best_index = rank[0]
    best_perf = np.max(perfs)

    return grouped_params[rank[0]], best_index, best_perf, perfs, rank

def plot_mean_ribbon(ax, mean, std, x_range, color=None, label=None, linestyle='solid'):
    line = ax.plot(x_range, mean, label=label, color=color, linestyle=linestyle)
    ax.fill_between(x_range, mean + std, mean - std, alpha=0.25, color=line[0].get_color())

