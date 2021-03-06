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


from src.analysis.learning_curve import plotBest
from src.experiment import ExperimentModel
from PyExpUtils.results.results import loadResults, whereParameterGreaterEq, whereParametersEqual, find
from PyExpUtils.utils.arrays import first
from tqdm import tqdm

# Mass taking imports from process_data for now
import os, sys, time, copy
sys.path.append(os.getcwd())
import numpy as np

from src.utils.json_handling import get_sorted_dict , get_param_iterable_runs
from src.utils.formatting import create_file_name, create_folder
from src.utils import analysis_utils

def _get_corner_loc(offsetx: int, offsety: int, loc_type: str):
    if (loc_type == 'center'):
        return [0.5 + offsetx, 0.5 + offsety]
    if (loc_type == 'top_left'):
        return [0.0 + offsetx, 0.0 + offsety]
    if (loc_type == 'top_right'):
        return [1.0 + offsetx, 0.0 + offsety]
    if (loc_type == 'bottom_left'):
        return [0.0 + offsetx, 1.0 + offsety]
    if (loc_type == 'bottom_right'):
        return [1.0 + offsetx, 1.0 + offsety]

# Returns a list of patch paths corresponding to each action in the q value
def _get_q_value_patch_paths(offsetx: int, offsety: int) -> list:
    center = _get_corner_loc(offsetx, offsety, 'center')
    top_left = _get_corner_loc(offsetx, offsety, 'top_left')
    top_right = _get_corner_loc(offsetx, offsety, 'top_right')
    bottom_left = _get_corner_loc(offsetx, offsety, 'bottom_left')
    bottom_right = _get_corner_loc(offsetx, offsety, 'bottom_right')

    top_action_path = [center, top_left, top_right, center]
    bottom_action_path = [center, bottom_left, bottom_right, center]
    right_action_path = [center, bottom_right, top_right, center]
    left_action_path = [center, bottom_left, top_left, center]

    action_path_map = [top_action_path, right_action_path, bottom_action_path, left_action_path]

    return action_path_map

def get_action_offset(magnitude: float):
    # UP RIGHT DOWN LEFT
    return [[0.0, -magnitude], [magnitude, 0.0], [0.0, magnitude], [-magnitude, 0.0]]

def get_text_location(offsetx:int, offsety:int, action: int):
    center = [0.5 + offsetx, 0.5 + offsety]
    text_offset_mag = 0.3
    text_offset = get_action_offset(text_offset_mag)
    x = center[0] + text_offset[action][0]
    y = center[1] + text_offset[action][1]
    return (x, y)

COLUMN_MAX = 12
ROW_MAX = 8

def _plot_init(ax, center_arrows: bool = False):
    ax.set_xlim(0, COLUMN_MAX)
    ax.set_ylim(0, ROW_MAX)
    ax.invert_yaxis()

    texts = []
    patches = []
    arrows = []
    for r in range(ROW_MAX):
        texts.append([])
        patches.append([])
        arrows.append([])
        for c in range(COLUMN_MAX):
            texts[r].append([])
            patches[r].append([])

            # Getting action triangle patches
            action_path_map = _get_q_value_patch_paths(c, r)

            for a in range(4):
                font = {
                    'size': 8
                }
                text_location = get_text_location(c, r ,a)
                # 'p' is just placeholder text
                texts[r][c].append(ax.text(text_location[0], text_location[1], 'p', fontdict = font, va='center', ha='center'))

                # placeholder color
                color = "blue"
                path = action_path_map[a]
                patch = ax.add_patch(PathPatch(Path(path), facecolor=color, ec='None'))
                patches[r][c].append(patch)

            # Getting default arrow. Making sure that this gets put on top of the patches
            center = [0.5 + r, 0.5 + c]
            if center_arrows:
                arrow = ax.arrow(center[0], center[1], 0.25, 0.25, width=0.025)
                arrows[r].append(arrow)

    
    if (center_arrows):
        return texts, patches, arrows
    
    return texts, patches

def scale_value(value: float, min_val:float, max_val:float):
    percentage = (value - min_val) / (max_val - min_val)
    percentage = np.cbrt(percentage)
    return percentage

def get_experiment_name():
    experiment_name = input("Give the input for experiment name (extension will be appended): ")

    while (len(experiment_name) == 0):
        experiment_name = input("Please enter an experiment name that is longer than 0 length: ")
    return experiment_name

def generatePlot(json_handle):
    data = load_experiment_data(json_handle)

    # print(return_data)
    # Processing data here so the dimensions are correct
    print(data.keys())
    data = data["update_distribution"]
    print(data.shape)
    data = data[:, 0, :, :]
    data = np.mean(data, axis=0)
    print(data.shape)

    # data = load_experiment_data(exp_path, file_name)

    experiment_name = get_experiment_name()

    anim_file_name = f'{experiment_name}_update_distribution.mp4'
    save_path = "./visualizations/"
    save_folder = os.path.splitext(save_path)[0]
    save_file = save_folder + f'/{anim_file_name}'

    if (not os.path.isdir(save_folder)):
        os.makedirs(save_folder)

    min_val = 0
    max_val = 1
    print(f'min: {min_val} max: {max_val}')

    fig, axes = plt.subplots(1, figsize=(16, 16))
    ax = axes

    colormap = cm.get_cmap('viridis')

    texts, patches, arrows = _plot_init(ax, center_arrows=True)
    
    # max_frames = 20
    # interval = 1
    start_frame = 0
    max_frame = 1000
    interval = 10
    frames = range(start_frame, max_frame, interval)

    print(f'Creating video from episode {start_frame} to episode {max_frame} at interval {interval}')
    pbar = tqdm(total=max_frame - start_frame)
    def draw_func(i):
        pbar.update(i - start_frame - pbar.n)
        q_values = np.sum(data[np.max([i - interval, 0]):i, :], axis=0)
        if np.linalg.norm(q_values, ord=1) != 0:
            q_values = q_values / np.linalg.norm(q_values, ord=1)
        else:
            print('np.linalg.norm errored out')
        # print(q_values.shape)

        ax.set_title(f"episode: {i}")

        for r in range(ROW_MAX):
            for c in range(COLUMN_MAX):
                q_value = [q_values[r * COLUMN_MAX + c]] * 4
                action = 2
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
                    scaled_value = scale_value(q_value[a], min_val, max_val)
                    patches[r][c][a].set_facecolor(colormap(scaled_value))
                    # colors = ["red", "green", "blue", "orange"]
                    # patches[i][j][a].set_facecolor(colors[a])
                    texts[r][c][a].set_text(round(q_value[a], 2))
        return

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

    # read the arguments etc
    if len(sys.argv) < 2:
        print("usage : python analysis/process_data.py <list of json files")
        exit()


    json_handle = get_json_handle()

    # Only use the first handle for now?
    generatePlot(json_handle)

    exit()