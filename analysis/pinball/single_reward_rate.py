
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
from analysis.common import get_best_grouped_param, load_data, load_reward_rate, load_max_reward_rate
from  experiment_utils.analysis_common.cache import cache_local_file
from pathlib import Path
from experiment_utils.analysis_common.colors import TOL_BRIGHT

STEP_SIZE = 20
# Plots the reward rate for a single run. Mainly used for debugging

def plot_single_reward_rate(ax, param_file_name: str, label: str=None, color=None):
    if label is None:
        label = Path(param_file_name).stem

    parameter_list = get_configuration_list_from_file_path(param_file_name)
    # print("plotting only the first index of the config list")

    index = 0

    ############ STANDARD
    data = load_data(parameter_list[index], 'reward_rate')

    # if param_file_name == 'experiments/pinball/scratch/scratch_gsp_learn.py':
    #     data = data[1000:]

    print(data.shape)
    run_data = mean_chunk_data(data, STEP_SIZE, 0)
    # print(run_data.shape)

    # # Accumulating
    # for i in range(1, len(run_data)):
    #     run_data[i] = run_data[i] + run_data[i - 1]

    x_range = get_x_range(0, run_data.shape[0], STEP_SIZE)

    # # print(len(list(x_range)))
    ax.plot(x_range, run_data, label=label, color=color)

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
    ax.set_xlabel('number of steps x100')
    ax.set_ylabel('reward rate')
    # plot_max_reward_rate(ax, 'experiments/chunlok/env_hmaze/GSP_no_direct.py')
    # plot_single_reward_rate(ax, 'experiments/chunlok/env_hmaze/2022_03_07_small_sweep/dyna.py')
    # plot_single_reward_rate(ax, 'experiments/chunlok/env_hmaze/dyna.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/impl_test.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/GSP_baseline.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/gsp_target.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/dqn.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/gsp_pretrain_test.py')

    # plot_single_reward_rate(ax, 'experiments/pinball/oracle_q_test.py')

    # plot_single_reward_rate(ax, 'experiments/pinball/oracle_q_test_only_values_single_goal_0.5.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/oracle_q_test_only_values_single_goal_1.0.py')

    # plot_single_reward_rate(ax, 'experiments/pinball/oracle_q_test_only_values_debug.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/oracle_q_test_only_values_debug_2.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/oracle_q_test_only_values_debug_3.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/oracle_q_test_only_values_debug_4.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/oracle_q_test_only_values_debug_5.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/oracle_q_test_pretrain_learn.py')


    # For pinball
    # plot_single_reward_rate(ax, 'experiments/pinball/oracle_q_learn.py', label='ddqn baseline')
    # plot_single_reward_rate(ax, 'experiments/pinball/oracle_q_test.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/suboptimal/suboptimal_pretrain_learn.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/suboptimal/subopti/mal_prefill_learn.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/suboptimal/suboptimal_prefill_debug.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/refactor/gsp_learn.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/refactor/gsp_learn_long.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/refactor/gsp_learn_long_debug.py', label='gsp baseline')
    # plot_single_reward_rate(ax, 'experiments/pinball/refactor/beta_gsp_learn_scratch_model_display.py', label = 'gsp online')
    # plot_single_reward_rate(ax, 'experiments/pinball/refactor/beta_gsp_learn_scratch_model_baseline_display.py', label = 'baseline gsp')

    # plot_single_reward_rate(ax, 'experiments/pinball/scratch/scratch_model_gsp_learn_100k.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/scratch/scratch_model_gsp_learn_200k.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/scratch/scratch_model_gsp_learn_300k.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/scratch/scratch_model_gsp_learn_final.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/scratch/scratch_model_gsp_learn_init.py')

    # plot_single_reward_rate(ax, 'experiments/pinball/scratch/scratch_gsp_learn_short.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/scratch/short_scratch_model_gsp_learn_10k.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/scratch/short_scratch_model_gsp_learn_25k.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/scratch/short_scratch_model_gsp_learn_50k.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/scratch/short_scratch_model_gsp_learn_75k.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/scratch/short_scratch_model_gsp_learn_final.py')


    list(TOL_BRIGHT.values())[0]
    # plot_single_reward_rate(ax, 'experiments/pinball/scratch/behaviour/short_scratch_model_gsp_learn_10k.py', label='gsp 10k model', color=list(TOL_BRIGHT.values())[1])
    # plot_single_reward_rate(ax, 'experiments/pinball/scratch/behaviour/short_scratch_model_gsp_learn_25k.py', label='gsp 25k model', color=list(TOL_BRIGHT.values())[1])
    # plot_single_reward_rate(ax, 'experiments/pinball/scratch/behaviour/short_scratch_model_gsp_learn_50k.py', label='gsp 50k model', color=list(TOL_BRIGHT.values())[2])
    # plot_single_reward_rate(ax, 'experiments/pinball/scratch/behaviour/short_scratch_model_gsp_learn_75k.py', label='gsp 75k model', color=list(TOL_BRIGHT.values())[3])
    # plot_single_reward_rate(ax, 'experiments/pinball/scratch/behaviour/short_scratch_model_gsp_learn_100k.py', label='gsp 100k model', color=list(TOL_BRIGHT.values())[4])
    
    # plot_single_reward_rate(ax, 'experiments/pinball/scratch/behaviour/short_scratch_model_gsp_learn_125k.py', label='gsp 125k model')
    # plot_single_reward_rate(ax, 'experiments/pinball/scratch/behaviour/short_scratch_model_gsp_learn_150k.py', label='gsp 150k model')
    # plot_single_reward_rate(ax, 'experiments/pinball/scratch/behaviour/short_scratch_model_gsp_learn_175k.py', label='gsp 175k model')
    # plot_single_reward_rate(ax, 'experiments/pinball/scratch/behaviour/short_scratch_model_gsp_learn_final.py', label='gsp 200k model')



    # plot_single_reward_rate(ax, 'experiments/pinball/refactor/gsp_learn_0.1.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/refactor/gsp_learn_0.1_online.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/refactor/gsp_learn_use_baseline.py')

    # Penalty env
    # plot_single_reward_rate(ax, 'experiments/pinball/penalty/dqn.py')

    # For hard pinball
    plot_single_reward_rate(ax, 'experiments/pinball_hard/baseline.py')
    # plot_single_reward_rate(ax, 'experiments/pinball_hard/oracle_gsp_learn.py')

    plot_single_reward_rate(ax, 'experiments/pinball_hard/sweep/gsp_learn_beta_ddqn_value_model_refactor_display.py', label = 'gsp 0.1')
    plot_single_reward_rate(ax, 'experiments/pinball_hard/sweep/baseline_display.py', label = 'baseline')
    
    # plot_single_reward_rate(ax, 'experiments/pinball/oracle_q_test_only_values.py')
    
    
    # plot_single_reward_rate(ax, 'experiments/pinball/dqn.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/gsp_pretrain_test.py')

    # plot_single_reward_rate(ax, 'experiments/pinball/GSP_learning_oracle.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/GSP_learning.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/GSP_learning_baseline.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/dreamer.py')
    # plot_single_reward_rate(ax, 'experiments/pinball/GSP_learning_values.py')

    

    # plot_single_reward_rate(ax, 'experiments/chunlok/env_hmaze/2022_03_07_small_sweep/dynaoptions.py')
    # plot_single_reward_rate(ax, 'experiments/chunlok/env_hmaze/2022_03_07_small_sweep/OCG.py')
    # plot_single_reward_rate(ax, 'experiments/chunlok/env_hmaze/2022_03_07_small_sweep/OCI.py')


    # plot_single_reward_rate(ax, 'experiments/chunlok/env_hmaze/dynaoptions.py')
    # plot_single_reward_rate(ax, 'experiments/chunlok/env_hmaze/OCI.py')
    # plot_single_reward_rate(ax, 'experiments/chunlok/env_hmaze/OCG.py')

    # Adaptation experiment
    # plot_single_reward_rate(ax, 'experiments/pinball/explore/gsp_learn.py', 'DDQN')
    # plot_single_reward_rate(ax, 'experiments/pinball/explore/gsp_learn_beta.py', 'GSP beta=0.1')

    plt.legend()
    # plt.title(alg_name)

    # ax.set_xlim([600, 1200])
    # ax.set_ylim(-10, 160)

    # Getting file name
    save_file = get_file_name('./plots/', f'single_reward_rate', 'png', timestamp=True)
    # print(f'Plot will be saved in {save_file}')
    plt.savefig(f'{save_file}', dpi = 300)
    
    # plt.show()

        