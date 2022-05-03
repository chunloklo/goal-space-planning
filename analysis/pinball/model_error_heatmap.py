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
from src.problems.PinballProblem import PinballOracleProblem, PinballProblem

from src.utils.log_utils import get_last_pinball_action_value_map

from src.experiment import ExperimentModel
import pickle


ROWS = 40
COLUMNS = ROWS
NUM_GOALS = 16

def generatePlots(data):
    # index will always be 0 since there's only 1 parameter there
    idx = 0

    fig, axes = plt.subplots(1, figsize=(40, 40))
    # axes = axes.flatten()

    save_file = get_file_name('./plots/', f'model_error_heatmap', 'png', timestamp=True)

    ax = axes


    colormap = cm.get_cmap('viridis')

    texts, patches = _plot_init(ax, columns = COLUMNS, rows = ROWS)

    # focus = True

    # if focus:
    #     min_val = np.min(data[g][initiation_map[g] == True])
    #     max_val = np.max(data[g][initiation_map[g] == True])
    # else:
    #     min_val = np.min(data[g])
    #     max_val = np.max(data[g])
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    
    # min_val = -50
    # max_val = 0
    print(min_val, max_val)
    # print(initiation_map[g].shape)
    
    for r in range(ROWS):
        for c in range(COLUMNS):
            try:
                state_data = data[r, c]
                # print(state_data.shape)

                a_map = {
                    0: 1,
                    1: 0,
                    2: 4,
                    3: 2,
                }

                for a in range(4):
                    # scaled_value = scale_value(state_data[a_map[a]], min_val, max_val, post_process_func=lambda x: x)

                    scaled_value = scale_value(state_data[a_map[a]], min_val, max_val)


                    # if focus:
                    #     if initiation_map[g, r, c] == True:
                    #         patches[r][c][a].set_facecolor(colormap(scaled_value))
                    #     else:
                    #         patches[r][c][a].set_facecolor((1, 1, 1))
                    # else:  
                    patches[r][c][a].set_facecolor(colormap(scaled_value))

                    texts[r][c][a].set_text(round(state_data[a_map[a]], 2))
            except KeyError as e:
                pass

    plt.savefig(save_file)
    plt.close()

if __name__ == "__main__":
    parameter_path = 'experiments/pinball/oracle_gsp_goal_model_learn.py'
    parameter_list = get_configuration_list_from_file_path(parameter_path)

    config = parameter_list[0]

    model_name = 'oracle_gsp_model_explore_4'
    goal_learner = pickle.load(open(f'./src/environments/data/pinball/{model_name}_goal_learner.pkl', 'rb'))

    behaviour_goal_value = 'q_learn'
    agent = pickle.load(open(f'src/environments/data/pinball/{behaviour_goal_value}_agent.pkl', 'rb'))
    behaviour_goal_value = agent.behaviour_learner

    goal_value_name = 'oracle_goal_values'
    goal_estimate_learner = pickle.load(open(f'./src/environments/data/pinball/{goal_value_name}_pretrain_goal_estimate_learner.pkl', 'rb'))
    goal_value_learner = pickle.load(open(f'./src/environments/data/pinball/{goal_value_name}_pretrain_goal_value_learner.pkl', 'rb'))
           

    exp_params = {
        'agent': config['agent'],
        'problem': config['problem'],
        'max_steps': 0,
        'episodes': 0,
        'metaParameters': config
    }


    exp = ExperimentModel.load_from_params(exp_params)
    problem = PinballOracleProblem(exp, 0, 0)

    def _get_behaviour_goal_values(xs, behaviour_goal_value, goal_initiation_func):
        batch_size = xs.shape[0]
        targets = np.array(behaviour_goal_value.get_action_values(xs))

        # Masking out invalid goals based on the initiation func
        for i in range(batch_size):
            x_goal_init = goal_initiation_func(xs[i])
            if np.all(~x_goal_init):
                targets[i] = np.nan

        return np.max(targets, axis=1)

    num_actions = problem.actions

    def goal_initiation(xs):
        init = problem.goals.goal_initiation(xs)
        # init[0] = False
        return init

    def get_error(s):
        batch_size = 1
        xs = np.array([s])
        num_goals = problem.goals.num_goals
        num_actions = problem.actions
        goal_states = np.hstack((problem.goals.goals, problem.goals.goal_speeds))
        # goal_dest_values = np.array(behaviour_goal_value.get_action_values(goal_states))
        # goal_dest_values = np.max(goal_dest_values, axis=1)

        goal_dest_values = goal_value_learner.goal_values

        goal_r = np.empty((batch_size, num_goals, num_actions))
        goal_gammas = np.empty((batch_size, num_goals, num_actions))
        # The goal policy is not used right now
        goal_policy_q = np.empty((batch_size, num_goals, num_actions)) 

        for g in range(num_goals):
            goal_policy_q[:, g, :], goal_r[:, g, :], goal_gammas[:, g, :] = goal_learner[g].get_goal_outputs(xs)
        pass

        # Getting one-hot policies for the goal policies
        goal_policies = np.zeros((batch_size, num_goals, num_actions))
        np.put_along_axis(goal_policies, np.expand_dims(np.argmax(goal_policy_q, axis=2), -1), 1, axis=2)

        goal_r = np.sum(goal_r * goal_policies, axis=2)
        goal_gammas = np.sum(goal_gammas * goal_policies, axis=2)

        goal_gammas = np.clip(goal_gammas, 0, 1)
        
        goal_values = goal_r + goal_gammas * goal_dest_values

        # Masking out invalid goals based on the initiation func
        for i in range(batch_size):
            x_goal_init = goal_initiation(xs[i])
            invalid_goals = np.where(x_goal_init == False)[0]
            goal_values[i, invalid_goals] = np.nan
        targets = np.nanmax(goal_values, axis=1)

        ##### Mainly checking for errors
        oracle_goal_values = _get_behaviour_goal_values(xs, behaviour_goal_value, goal_initiation)

        return np.nanmean(np.square(oracle_goal_values - targets))

    RESOLUTION = 40
    last_goal_q_map = np.zeros((RESOLUTION, RESOLUTION, num_actions))
    goal_action_value = get_last_pinball_action_value_map(1, get_error, resolution=RESOLUTION)
    last_goal_q_map = goal_action_value[0]
    data = last_goal_q_map
    # print(data.shape)

    generatePlots(data)

    exit()