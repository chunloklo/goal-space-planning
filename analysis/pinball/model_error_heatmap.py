from functools import partial
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
from src.problems.PinballProblem import PinballHardProblem, PinballOracleProblem, PinballProblem, PinballSuboptimalProblem
from src.problems.registry import getProblem
from src.utils.log_utils import get_last_pinball_action_value_map

from src.experiment import ExperimentModel
import pickle


ROWS = 80
COLUMNS = ROWS
NUM_GOALS = 16

def generatePlots(data, label):
    # index will always be 0 since there's only 1 parameter there
    idx = 0

    fig, axes = plt.subplots(1, figsize=(40, 40))
    # axes = axes.flatten()

    save_file = get_file_name('./plots/', f'{label}_model_error_heatmap', 'png', timestamp=True)

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

    # min_val = 0
    # max_val = 0.3
    
    # min_val = -50
    # max_val = 0
    # print(min_val, max_val)
    # print(initiation_map[g].shape)
    
    for r in range(ROWS):
        for c in range(COLUMNS):
            try:
                state_data = data[r, c]

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
    # parameter_path = 'experiments/pinball/refactor/goal_model_learn_long.py'
    parameter_path = 'experiments/pinball/scratch/short_scratch_model_gsp_learn_final.py'
    parameter_list = get_configuration_list_from_file_path(parameter_path)

    config = parameter_list[0]
    # Modifying configs
    config['explore_env'] = False

    model_name = config['load_model_name']
    # model_name = 'pinball_refactor_eps'
    print(f'model name: {model_name}')
    goal_learner = pickle.load(open(f'./src/environments/data/pinball/{model_name}_goal_learner.pkl', 'rb'))

    # sdfsd
    exp_params = {
        'agent': config['agent'],
        'problem': config['problem'],
        'max_steps': 0,
        'episodes': 0,
        'metaParameters': config
    }

    num_actions = 5

    exp = ExperimentModel.load_from_params(exp_params)
    problem = getProblem(config['problem'])(exp, 0, 0)
    gamma = config['gamma']

    env = problem.env
    env.start()

    init_func = problem.goals.goal_initiation
    term_func = problem.goals.goal_termination

    def get_error(s, g):
        if not init_func(s)[g]:
            return np.nan

        # xs = np.array([s])
        goal_policy , goal_r, goal_gammas = goal_learner[g].get_goal_outputs(np.array([s]))

        goal_action = np.argmax(goal_policy)


        r_s = goal_r[0][goal_action]
        gamma_s = goal_gammas[0][goal_action]

        # return gamma_s

        # Rolling out in the env
        env.pinball.ball.position = np.copy(s[:2])
        env.pinball.ball.xdot = 0.0
        env.pinball.ball.ydot = 0.0

        colliding = False
        for obs in problem.env.pinball.obstacles:
            if obs.collision(problem.env.pinball.ball):
                colliding = True
                break
        
        if colliding:
            return np.nan

        # Starting
        roll_s = env.pinball.get_state()
        # print(roll_s)
        discount = 1
        discounted_reward = 0

        goal_terminated = True
        for _ in range(60):
            goal_policy, _, _ = goal_learner[g].get_goal_outputs(np.array([roll_s]))
            roll_a = int(np.argmax(goal_policy))
            # print(roll_a)
            # print(f'{goal_policy} {roll_a}')


            (reward, next_state, termination), info = env.step(roll_a)
            discounted_reward += discount * reward
            discount *= gamma

            # If you go outside the area, then its nan
            # print(next_state)
            next_state = np.array(next_state)

            if term_func(roll_s, roll_a, next_state)[g]:
                # goal_terminated = True
                break

            if not init_func(next_state)[g]:
                return np.nan

            roll_s = next_state

        if not goal_terminated:
            # print('did not terminate from rollout')
            return np.nan
        # print(f's:{s}')
        # print(f'gamma: {gamma_s} discount {discount}, error {np.square(gamma_s - discount)}')

        # return discount
        return np.abs(gamma_s - discount)
        # return np.abs(r_s - discounted_reward)

    RESOLUTION = ROWS
    last_goal_q_map = np.zeros((RESOLUTION, RESOLUTION, num_actions))
    g = 2
    goal_action_value = get_last_pinball_action_value_map(1, partial(get_error, g=g), resolution=RESOLUTION, show_progress=True)
    last_goal_q_map = goal_action_value[0]
    data = last_goal_q_map

    # print(data.shape)
    # print(data[10, 30, :])
    # sdfsd

    print('Data generated')
    generatePlots(data, g)

    exit()