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
import matplotlib


from src.analysis.learning_curve import plotBest
from src.experiment import ExperimentModel
from PyExpUtils.utils.arrays import first
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
import argparse
import importlib.util
from src.utils import run_utils
from collections.abc import Iterable

COLUMN_MAX = 12
ROW_MAX = 8

def generatePlot(data):

    # print("backend", plt.rcParams["backend"])
    # For now, we can't use the default MacOSX backend since it will give me terrible corruptions
    matplotlib.use("TkAgg")

    # Processing data here so the dimensions are correct
    print(data.keys())
    data = data["Q"]
    data = np.array(data)
    print(data.shape)
    # print(data)
    data = data[0, :, :, :]
    # data = np.mean(data, axis=0)
    print(data.shape)

    # Getting file name
    save_file = prompt_user_for_file_name('./visualizations/', 'action_values_', '', 'mov', timestamp=True)
    # print(f'Plot will be saved in {anim_file_name}')

    print(f'Visualization will be saved in {save_file}')
    # Getting episode range
    start_frame, max_frame, interval = prompt_episode_display_range(0, data.shape[0], max(data.shape[0] // 100, 1))

    # Using a simple way of determining whether options are used.
    # Note that this might not work in the future if we do something separate, but it works for now
    hasOptions = data.shape[-1] > 4

    num_options = data.shape[-1] - 4

    # fig = plt.figure()
    if hasOptions:
        fig, axes = plt.subplots(1, 2, figsize=(32, 16))
        ax = axes[0]
        ax_options = axes[1]
        
    else:
        fig, axes = plt.subplots(1, figsize=(16, 16))
        ax = axes

    colormap = cm.get_cmap('viridis')

    texts, patches, arrows = _plot_init(ax, columns = COLUMN_MAX, rows = ROW_MAX, center_arrows=True)
    

    if hasOptions:
        ax_options.set_xlim(0, COLUMN_MAX)
        ax_options.set_ylim(0, ROW_MAX)
        ax_options.invert_yaxis()
        texts_options, patches_options = _plot_init(ax_options, columns = COLUMN_MAX, rows = ROW_MAX, center_arrows=False)
    
    wall_indices = [12, 14, 24, 25, 26, 30, 32, 42, 43, 44 ]

    min_val = np.min(np.delete(data[start_frame:max_frame], wall_indices, axis=1))
    max_val = np.max(data[start_frame:max_frame])
    print(f'min: {min_val} max: {max_val}')


    frames = range(start_frame, max_frame, interval)

    x_range = list(get_x_range(0, data.shape[0], 1))

    print(f'Creating video from episode {start_frame} to episode {max_frame} at interval {interval}')
    pbar = tqdm(total=max_frame - start_frame)

    # return [*flatten(texts), *flatten(patches), *flatten(arrows), *flatten(texts_options), *flatten(patches_options)]
    def draw_func(i):
        pbar.update(i - start_frame - pbar.n)
        q_values = data[i, :, :]
        # print(q_values)


        ax.set_title(f"episode: {x_range[i]}")

        for r in range(ROW_MAX):
            for c in range(COLUMN_MAX):
                q_value = q_values[r * COLUMN_MAX + c, :]
                action = np.argmax(q_value)
                arrow_magnitude = 0.0625
                width = 0.025
                center = [0.5 + c, 0.5 + r]
                offset = get_action_offset(arrow_magnitude)

                arrows[r][c].remove()

                if (action < 4):
                    offset = get_action_offset(arrow_magnitude)
                    arrow = ax.arrow(center[0], center[1], offset[action][0], offset[action][1], width=width, facecolor='black')
                    arrows[r][c] = arrow
                else:
                    option = action - 4

                    from src.utils.create_options import get_options
                    options = get_options('GrazingAdam')
                    action, _ = options[option].step(r * COLUMN_MAX + c)

                    offset = get_action_offset(arrow_magnitude)
                    arrow = ax.arrow(center[0], center[1], offset[action][0], offset[action][1], width=width, facecolor='red')
                    arrows[r][c] = arrow

                for a in range(4):
                    scaled_value = scale_value(q_value[a], min_val, max_val, post_process_func=lambda x: x)
                    patches[r][c][a].set_facecolor(colormap(scaled_value))
                    # colors = ["red", "green", "blue", "orange"]
                    # patches[i][j][a].set_facecolor(colors[a])
                    texts[r][c][a].set_text(round(q_value[a], 2))

                if hasOptions:
                    for a in range(num_options):
                        scaled_value = scale_value(q_value[a + 4], min_val, max_val)
                        patches_options[r][c][a].set_facecolor(colormap(scaled_value))
                        texts_options[r][c][a].set_text(round(q_value[a + 4], 2))

        return

    animation = FuncAnimation(fig, draw_func, frames=frames, blit=False)
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

    # Parsing arguments
    parser = argparse.ArgumentParser(description='Produces the action values video for GrazingWorld Adam')
    parser.add_argument('parameter_path', help='path to the Python parameter file that contains a get_configuration_list function that returns a list of parameters to run')
    args = parser.parse_args()

    parameter_path = args.parameter_path

    # Getting parameter list from parameter_path
    param_spec = importlib.util.spec_from_file_location("ParamModule", parameter_path)
    ParamModule = importlib.util.module_from_spec(param_spec)
    param_spec.loader.exec_module(ParamModule)
    parameter_list = ParamModule.get_configuration_list()

    parameter_list = list(filter(lambda x: x['alpha']==1.0, parameter_list))

    data = run_utils.load_data(parameter_list[0])
    # print(parameter_list[0])
    # print(parameter_list[10])
    # adas

    generatePlot(data)

    exit()