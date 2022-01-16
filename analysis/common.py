from typing import Any, Callable, List, Tuple
from analysis_common.configs import group_configs
import numpy as np
from src.utils import run_utils

# Some common functions for analysis specific for this project

# You can likely cache this for better performance later.
def load_reward_rate(config):
    data = run_utils.load_data(config)

    # Just getting rid of the first dimension
    data = np.array(data['reward_rate'])
    return data[0]

def load_max_reward_rate(config):
    data = run_utils.load_data(config)
    # 1 x num_logs
    data = np.array(data['max_reward_rate'])

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
    perfs = [np.mean([get_performance(config) for config in config_list]) for _, config_list in grouped_params]

    best_index = np.argmax(perfs)
    best_perf = np.max(perfs)

    return grouped_params[best_index], best_index, best_perf, perfs

def plot_mean_ribbon(ax, mean, std, x_range, color=None, label=None):
    line = ax.plot(x_range, mean, label=label, color=color)
    ax.fill_between(x_range, mean + std, mean - std, alpha=0.25, color=line[0].get_color())

