from genericpath import isdir
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib import cm
from matplotlib.animation import FuncAnimation
sys.path.append(os.getcwd())

from src.analysis.learning_curve import plotBest
from src.experiment import ExperimentModel
from PyExpUtils.results.results import loadResults, whereParameterGreaterEq, whereParametersEqual, find
from PyExpUtils.utils.arrays import first
from tqdm import tqdm

def _get_corner_loc(offsetx: int, offsety: int, loc_type: str):
    if (loc_type == 'center'):
        return [0.5 + offsetx, 0.5 + offsety]
    if (loc_type == 'top_left'):
        return [0.0 + offsetx, 1.0 + offsety]
    if (loc_type == 'top_right'):
        return [1.0 + offsetx, 1.0 + offsety]
    if (loc_type == 'bottom_left'):
        return [0.0 + offsetx, 0.0 + offsety]
    if (loc_type == 'bottom_right'):
        return [1.0 + offsetx, 0.0 + offsety]

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

    action_path_map = [left_action_path, right_action_path, top_action_path, bottom_action_path]

    return action_path_map

def get_text_location(offsetx:int, offsety:int, action: int):
    center = [0.5 + offsetx, 0.5 + offsety]
    text_offset_mag = 0.3
    text_offset = [[-text_offset_mag, 0], [text_offset_mag, 0.0], [0.0, text_offset_mag], [0.0, -text_offset_mag]]
    x = center[0] + text_offset[action][0]
    y = center[1] + text_offset[action][1]
    return (x, y)

def _plot_init(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.invert_yaxis()

    texts = []
    patches = []
    for i in range(10):
        texts.append([])
        patches.append([])
        for j in range(10):
            texts[i].append([])
            patches[i].append([])

            action_path_map = _get_q_value_patch_paths(i, j)
                
            
            text_offset_mag = 0.3
            text_offset = [[-text_offset_mag, 0], [text_offset_mag, 0.0], [0.0, text_offset_mag], [0.0, -text_offset_mag]]

            for a in range(4):
                font = {
                    'size': 8
                }
                text_location = get_text_location(i ,j ,a)
                # 'p' is just placeholder text
                texts[i][j].append(ax.text(text_location[0], text_location[1], 'p', fontdict = font, va='center', ha='center'))

                # placeholder color
                color = "blue"
                path = action_path_map[a]
                patch = ax.add_patch(PathPatch(Path(path), facecolor=color, ec='None'))
                patches[i][j].append(patch)
    
    return texts, patches


def scale_value(value: float, min_val:float, max_val:float):
    percentage = (value - min_val) / (max_val - min_val)
    percentage = np.cbrt(percentage)
    return percentage

def load_experiment_data(exp_path, file_name):
    exp = ExperimentModel.load(exp_path)
    results = loadResults(exp, file_name)
    data = None
    for r in results:
        data = r.load()
    return data

def generatePlot(exp_paths, file_name, anim_file_name):
    for exp_path in exp_paths:
        data = load_experiment_data(exp_path, file_name)

        save_path = exp_path.replace('experiments', 'visualizations')
        save_folder = os.path.splitext(save_path)[0]
        save_file = save_folder + f'/{anim_file_name}'

        if (not os.path.isdir(save_folder)):
            os.makedirs(save_folder)

        # Using a simple way of determining whether options are used.
        # Note that this might not work in the future if we do something separate, but it works for now
        hasOptions = data.shape[-1] > 4

        min_val = np.min(data)
        max_val = np.max(data)
        print(f'min: {min_val} max: {max_val}')

        # fig = plt.figure()
        if hasOptions:
            fig, axes = plt.subplots(1, 2, figsize=(32, 16))
            ax = axes[0]
            ax_options = axes[1]
            
        else:
            fig, axes = plt.subplots(1, figsize=(16, 16))
            ax = axes

        colormap = cm.get_cmap('viridis')

        texts, patches = _plot_init(ax)
        

        if hasOptions:
            ax_options.set_xlim(0, 10)
            ax_options.set_ylim(0, 10)
            ax_options.invert_yaxis()
            texts_options, patches_options = _plot_init(ax_options)

        pbar = tqdm(total=data.shape[0])
        def draw_func(i):
            pbar.update(i - pbar.n)
            q_values = data[i, :, :]

            # titles.append(ax.set_title(f"frame: {i}"))
            ax.set_title(f"frame: {i}")

            for i in range(10):
                for j in range(10):
                    q_value = q_values[i + j * 10, :]
                    for a in range(4):
                        scaled_value = scale_value(q_value[a], min_val, max_val)
                        patches[i][j][a].set_facecolor(colormap(scaled_value))
                        texts[i][j][a].set_text(round(q_value[a], 2))

                    if hasOptions:
                        for a in range(3):
                            scaled_value = scale_value(q_value[a + 4], min_val, max_val)
                            patches_options[i][j][a].set_facecolor(colormap(scaled_value))
                            texts_options[i][j][a].set_text(round(q_value[a + 4], 2))
            return

        animation = FuncAnimation(fig, draw_func, frames=range(0, data.shape[0], 5))
        animation.save(save_file)
        pbar.close()
        # plt.show()

if __name__ == "__main__":
    exp_paths = sys.argv[1:]
    generatePlot(exp_paths, 'Q.npy', 'action_values.mp4')

    exit()