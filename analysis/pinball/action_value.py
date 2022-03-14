import os
import sys
sys.path.append(os.getcwd())

from genericpath import isdir
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib import cm
from matplotlib.animation import FuncAnimation

from tqdm import tqdm

# Mass taking imports from process_data for now
import os, sys, time, copy
sys.path.append(os.getcwd())
import numpy as np

from src.utils.json_handling import get_sorted_dict , get_param_iterable_runs
from src.utils.formatting import create_file_name, create_folder
from src.utils import analysis_utils
from src.analysis.gridworld_utils import _get_corner_loc, _get_q_value_patch_paths, get_file_name, get_text_location, prompt_user_for_file_name, get_action_offset, scale_value, _plot_init, prompt_episode_display_range
from src.analysis.plot_utils import get_x_range
from src.environments.HMaze import HMaze

from experiment_utils.sweep_configs.common import get_configuration_list_from_file_path
from src.analysis.plot_utils import load_configuration_list_data
from analysis.common import get_best_grouped_param, load_reward_rate, load_max_reward_rate
from  experiment_utils.analysis_common.configs import group_configs
from src.utils import run_utils
from analysis.common import load_data
import matplotlib
from datetime import datetime
from src.problems.PinballProblem import PinballProblem

from src.experiment import ExperimentModel


ROWS = 20
COLUMNS = 20
NUM_GOALS = 16

def generatePlots(param, data, key):
    time_str = datetime.now().strftime("%m-%d-%Y--%H-%M-%S")
    # folder = f'./plots/{key}_{time_str}'
    # os.makedirs(folder)

    exp_params = {
        'agent': param['agent'],
        'problem': param['problem'],
        'max_steps': 0,
        'episodes': 0,
        'metaParameters': param
    }
    # index will always be 0 since there's only 1 parameter there
    idx = 0

    exp = ExperimentModel.load_from_params(exp_params)


    problem = PinballProblem(exp, 0, 0)

    # Calculating the value at each state approximately
    num_goals = problem.num_goals
    initiation_map = np.zeros((num_goals, 20, 20))
    for r, y in enumerate(np.linspace(0, 1, 20)):
        for c, x in enumerate(np.linspace(0, 1, 20)):
            init = problem.goal_initiation_func([x, y, 0, 0])
            initiation_map[:, r, c] = init

    fig, ax = plt.subplots(1, figsize=(16, 16))

    save_file = get_file_name('./plots/', f'{key}', 'png', timestamp=True)
    ax.set_title

    colormap = cm.get_cmap('viridis')

    texts, patches = _plot_init(ax, columns = COLUMNS, rows = ROWS)

    min_val = np.min(data)
    max_val = np.max(data)
    
    for r in range(ROWS):
        for c in range(COLUMNS):
            try:
                state_data = data[r, c]
                # print(state_data.shape)

                a_map = {
                    0: 1,
                    1: 0,
                    2: 3,
                    3: 2,
                }

                for a in range(4):
                    scaled_value = scale_value(state_data[a_map[a]], min_val, max_val, post_process_func=lambda x: x)
                    patches[r][c][a].set_facecolor(colormap(scaled_value))
                    # colors = ["red", "green", "blue", "orange"]
                    # patches[i][j][a].set_facecolor(colors[a])
                    texts[r][c][a].set_text(round(state_data[a_map[a]], 2))
            except KeyError as e:
                pass

    plt.savefig(save_file)
    plt.close()

if __name__ == "__main__":
    parameter_path = 'experiments/pinball/impl_test.py'
    parameter_list = get_configuration_list_from_file_path(parameter_path)

    config = parameter_list[0]
    key = 'q_map'
    data = load_data(config, key)
    data = data.squeeze()
    print(data.shape)
    

    generatePlots(config, data, key)

    exit()