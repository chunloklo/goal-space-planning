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
                texts[i][j].append(ax.text(text_location[0], text_location[1], 'p', fontdict = font))

                # placeholder color
                color = "blue"
                path = action_path_map[a]
                patch = ax.add_patch(PathPatch(Path(path), facecolor=color, ec='None'))
                patches[i][j].append(patch)
    
    return texts, patches


def scale_value(value: float, min_val:float, max_val:float):
    percentage = (value + min_val) / (max_val - min_val)
    return percentage

def generatePlot(ax, exp_paths, bounds):
    for exp_path in exp_paths:
        exp = ExperimentModel.load(exp_path)

        results = loadResults(exp, 'Q.npy')
        data = None
        for r in results:
            data = r.load()

        print(data.shape)

        min_val = np.min(data)
        max_val = np.max(data)

        # fig = plt.figure()
        fig, ax = plt.subplots(1, figsize=(16, 16))

        colormap = cm.get_cmap('viridis')

        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.invert_yaxis()

        texts, patches = _plot_init(ax)

        def draw_func(i):
        # for now lets get the last one
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
            return

        animation = FuncAnimation(fig, draw_func, frames=range(0, data.shape[0], 1))
        animation.save('visualizations/option_q_action_values.mp4')
        # plt.show()

if __name__ == "__main__":
    # f, axes = plt.subplots(1)
    axes = None
    bounds = []

    exp_paths = sys.argv[1:]

    generatePlot(axes, exp_paths, bounds)

    # plt.show()
    exit()

    save_path = 'experiments/exp/plots'
    os.makedirs(save_path, exist_ok=True)

    width = 8
    height = (24/5)
    f.set_size_inches((width, height), forward=False)
    plt.savefig(f'{save_path}/learning-curve.png', dpi=100)
