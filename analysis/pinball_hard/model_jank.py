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
import cloudpickle


ROWS = 80
COLUMNS = ROWS
NUM_GOALS = 16

def generatePlots(data, label):
    # index will always be 0 since there's only 1 parameter there
    idx = 0

    fig, axes = plt.subplots(1, figsize=(40, 40))
    # axes = axes.flatten()

    save_file = get_file_name('./plots/', f'{label}_model_value_heatmap', 'png', timestamp=True)

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

def load_model(model_name):
    goal_learners = pickle.load(open(f'./src/environments/data/pinball/{model_name}_goal_learner.pkl', 'rb'))
    goal_buffers = pickle.load(open(f'./src/environments/data/pinball/{model_name}_goal_buffer.pkl', 'rb'))
    goal_estimate_buffer = pickle.load(open(f'./src/environments/data/pinball/{model_name}_goal_estimate_buffer.pkl', 'rb'))
    return goal_learners, goal_buffers, goal_estimate_buffer

def save_model(save_model_name, goal_learners, goal_buffers, goal_estimate_buffer):
    cloudpickle.dump(goal_learners, open(f'./src/environments/data/pinball/{save_model_name}_goal_learner.pkl', 'wb'))
    cloudpickle.dump(goal_buffers, open(f'./src/environments/data/pinball/{save_model_name}_goal_buffer.pkl', 'wb'))
    cloudpickle.dump(goal_estimate_buffer, open(f'./src/environments/data/pinball/{save_model_name}_goal_estimate_buffer.pkl', 'wb'))
    
if __name__ == "__main__":

    goal_learners, goal_buffers, goal_estimate_buffer = load_model('pinball_hard_ddqn_value_policy_test')
    goal_learners_8, goal_buffers_8, goal_estimate_buffer_8 = load_model('pinball_hard_ddqn_value_policy_add_test')
    goal_learners_7, goal_buffers_7, goal_estimate_buffer_7 = load_model('pinball_hard_ddqn_value_policy_add_test_debug')

    
    goal_learners.append(goal_learners_7[7])
    goal_learners.append(goal_learners_8[8])

    goal_buffers.append(goal_buffers_7[7])
    goal_buffers.append(goal_buffers_8[8])

    save_model('pinball_hard_ddqn_value_policy_8_goals', goal_learners, goal_buffers, goal_estimate_buffer_7)

    
exit()  