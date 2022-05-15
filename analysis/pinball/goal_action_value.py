from functools import partial
import os
import pickle
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
from src.problems.PinballProblem import PinballHardProblem, PinballOracleProblem, PinballProblem
from src.problems.registry import getProblem

from src.utils.log_utils import get_last_pinball_action_value_map

from src.experiment import ExperimentModel


ROWS = 40
COLUMNS = ROWS

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

    problem = getProblem(param['problem'])(exp, 0, 0)
    # problem = PinballProblem(exp, 0, 0)

    # Calculating the value at each state approximately
    num_goals = problem.goals.num_goals
    initiation_map = np.zeros((num_goals, ROWS, COLUMNS))
    for r, y in enumerate(np.linspace(0, 1, ROWS)):
        for c, x in enumerate(np.linspace(0, 1, ROWS)):
            init = problem.goals.goal_initiation([x, y, 0, 0])
            initiation_map[:, r, c] = init

    fig, axes = plt.subplots(3, 3, figsize=(90, 90))
    axes = axes.flatten()

    save_file = get_file_name('./plots/', f'{key}', 'png', timestamp=True)

    
    for g in tqdm(range(num_goals)):
        # print("backend", plt.rcParams["backend"])
        # For now, we can't use the default MacOSX backend since it will give me terrible corruptions
        # matplotlib.use("TkAgg")

        # fig, axes = plt.subplots(1, figsize=(16, 16))
        ax = axes[g]
        ax.set_title(g)

        colormap = cm.get_cmap('viridis')

        texts, patches = _plot_init(ax, columns = COLUMNS, rows = ROWS)

        min_val = np.min(data[g][initiation_map[g] == True])
        max_val = np.max(data[g][initiation_map[g] == True])
        print(min_val, max_val)
        
        for r in range(ROWS):
            for c in range(COLUMNS):
                try:
                    state_data = data[g, r, c]
                    # print(state_data.shape)

                    a_map = {
                        0: 1,
                        1: 0,
                        2: 3,
                        3: 2,
                    }

                    for a in range(4):
                        scaled_value = scale_value(state_data[a_map[a]], min_val, max_val, post_process_func=lambda x: x)

                        if initiation_map[g, r, c] == True:
                            patches[r][c][a].set_facecolor(colormap(scaled_value))
                        else:
                            patches[r][c][a].set_facecolor((1, 1, 1))
                        # colors = ["red", "green", "blue", "orange"]
                        # patches[i][j][a].set_facecolor(colors[a])
                        texts[r][c][a].set_text(round(state_data[a_map[a]], 2))
                except KeyError as e:
                    pass

    plt.savefig(save_file)
    plt.close()

if __name__ == "__main__":
    parameter_path = 'experiments/pinball/refactor/goal_model_learn.py'
    parameter_list = get_configuration_list_from_file_path(parameter_path)

    config = parameter_list[0]

    # model_name = config['save_model_name']
    model_name = 'pinball_scratch_model'
    goal_learners = pickle.load(open(f'./src/environments/data/pinball/{model_name}_goal_learner.pkl', 'rb'))

    print(model_name)
    print(f'num goal learners: {len(goal_learners)}')


    # sdfsd
    exp_params = {
        'agent': config['agent'],
        'problem': config['problem'],
        'max_steps': 0,
        'episodes': 0,
        'metaParameters': config
    }


    exp = ExperimentModel.load_from_params(exp_params)
    problem = getProblem(config['problem'])(exp, 0, 0)
    print(problem.goals.num_goals)

    def get_goal_outputs(s, g):
            action_value, reward, gamma = goal_learners[g].get_goal_outputs(s)
            return np.vstack([action_value, reward, gamma])

    RESOLUTION = ROWS
    last_goal_q_map = np.zeros((problem.goals.num_goals, RESOLUTION, RESOLUTION, 5))
    last_reward_map = np.zeros((problem.goals.num_goals, RESOLUTION, RESOLUTION, 5))
    last_gamma_map = np.zeros((problem.goals.num_goals, RESOLUTION, RESOLUTION, 5))
    
    for g in range(problem.goals.num_goals):
        goal_action_value = get_last_pinball_action_value_map(3, partial(get_goal_outputs, g=g), resolution=RESOLUTION)
        last_goal_q_map[g] = goal_action_value[0]
        last_reward_map[g] = goal_action_value[1]
        last_gamma_map[g] = goal_action_value[2]

    all_data = {
        'goal_r_map': last_reward_map,
        'goal_gamma_map': last_gamma_map,
    }

    key  = 'goal_r_map'
    data = all_data[key]
    print(f'data_shape: {data.shape}')
    # sdfsd
    generatePlots(config, data, key)

    exit()