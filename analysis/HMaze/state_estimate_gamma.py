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
from src.analysis.gridworld_utils import _get_corner_loc, _get_q_value_patch_paths, get_text_location, prompt_user_for_file_name, get_action_offset, scale_value, _plot_init, prompt_episode_display_range
from src.analysis.plot_utils import get_x_range
from src.environments.HMaze import HMaze

from experiment_utils.sweep_configs.common import get_configuration_list_from_file_path
from src.analysis.plot_utils import load_configuration_list_data
from analysis.common import get_best_grouped_param, load_reward_rate, load_max_reward_rate
from  experiment_utils.analysis_common.configs import group_configs
from src.utils import run_utils
from analysis.common import load_data
import matplotlib

def generatePlot(json_handle):
    # print("backend", plt.rcParams["backend"])
    # For now, we can't use the default MacOSX backend since it will give me terrible corruptions
    matplotlib.use("TkAgg")


    # Getting file name
    save_file = prompt_user_for_file_name('./visualizations/', 'state_estimate_r_', '', 'mov', timestamp=True)

    print(f'Visualization will be saved in {save_file}')
    # Getting episode range
    start_frame, max_frame, interval = prompt_episode_display_range(0, data.shape[0], max(data.shape[0] // 100, 1))

    env = HMaze(0)
    tab_feature = env.get_tabular_feature()

    fig, axes = plt.subplots(1, figsize=(16, 16))
    ax = axes

    colormap = cm.get_cmap('viridis')

    texts, patches = _plot_init(ax, columns = env.size, rows = env.size)
    

    min_val = np.min(data)
    max_val = np.max(data)
    print(f'min: {min_val} max: {max_val}')

    frames = range(start_frame, max_frame, interval)
    x_range = list(get_x_range(0, data.shape[0], 1))

    print(f'Creating video from episode {start_frame} to episode {max_frame} at interval {interval}')
    pbar = tqdm(total=max_frame - start_frame)

    plot_option = 1
    def draw_func(i):
        pbar.update(i - start_frame - pbar.n)
        frame_data = data[i]


        ax.set_title(f"episode: {x_range[i]}")
        nonzero_states = set()

        for r in range(env.size):
            for c in range(env.size):
                try:
                    state_index = tab_feature.encode((r, c))
                    # q_value = q_values[index, :]

                    state_data = frame_data[state_index, :, plot_option]
                    # print(state_data)
                    # q_value = q_values[index, option][[119, 120, 121]]

                    # _, states = np.nonzero(q_value)
                    # for s in states:
                    #     nonzero_states.add(s)
                    # print(np.nonzero(q_value))

                    for a in range(4):
                        scaled_value = scale_value(state_data[a], min_val, max_val, post_process_func=lambda x: x)
                        patches[r][c][a].set_facecolor(colormap(scaled_value))
                        # colors = ["red", "green", "blue", "orange"]
                        # patches[i][j][a].set_facecolor(colors[a])
                        texts[r][c][a].set_text(round(state_data[a], 2))
                except KeyError as e:
                    pass

    animation = FuncAnimation(fig, draw_func, frames=frames)
    animation.save(save_file)
    pbar.close()
    # plt.show()


def get_json_handle():
    json_files = sys.argv[1:] # all the json files
    # convert all json files to dict
    json_handles = [get_sorted_dict(j) for j in json_files]

    # Logic for choosing which json handle
    print("grabbing only the first experiment for visualization")
    return json_handles[0]

def load_experiment_data(json_handle, load_keys: list = None):
    # if load_keys is None, then it loads all the keys
    iterables = get_param_iterable_runs(json_handle)
        
    for i in iterables:
        print(i)
        return_data = analysis_utils.load_different_runs_all_data(i, load_keys)
        print(return_data.keys())
        # mean_return_data, stderr_return_data = process_runs(return_data)
        pass

    # Messy right now, but its okay
    return return_data


if __name__ == "__main__":
    parameter_path = 'experiments/chunlok/mpi/hmaze/optionplanning_test.py'
    parameter_list = get_configuration_list_from_file_path(parameter_path)



    data = load_data(parameter_list[0], 'state_estimate_gamma')

    print(data.shape)

    generatePlot(data)

    exit()