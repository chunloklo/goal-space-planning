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

from action_values import _plot_init, scale_value, get_json_handle, load_experiment_data, get_experiment_name


def generatePlot(json_handle):
    data = load_experiment_data(json_handle)

    # print(return_data)
    # Processing data here so the dimensions are correct
    data = data["model_r"]
    print(data.shape)
    data = data[:, 0, :, :, :]
    data = np.mean(data, axis=0)

    experiment_name = get_experiment_name()
    anim_file_name = f'{experiment_name}_model_r.mp4'

    save_path = "./visualizations/"
    save_folder = os.path.splitext(save_path)[0]
    save_file = save_folder + f'/{anim_file_name}'

    print(f'data shape: {data.shape}')

    min_val = np.min(data)
    max_val = np.max(data)

    print(f'min: {min_val} max: {max_val}')

    fig, axes = plt.subplots(1, figsize=(16, 16))
    ax = axes

    colormap = cm.get_cmap('viridis')

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.invert_yaxis()
    texts, patches = _plot_init(ax)

    num_options = data.shape[-1] 

    # max_frames = 50
    # interval = 1
    max_frames = data.shape[0]
    interval = 10
    frames = range(0, max_frames, interval)
    pbar = tqdm(total=max_frames)
    def draw_func(i):
        pbar.update(i - pbar.n)
        q_values = data[i, :, :]

        # titles.append(ax.set_title(f"frame: {i}"))
        ax.set_title(f"frame: {i}")

        for i in range(10):
            for j in range(10):
                q_value = q_values[i + j * 10, :]
                # print(q_values.shape)
                for a in range(num_options):
                    scaled_value = scale_value(q_value[a], min_val, max_val)
                    patches[i][j][a].set_facecolor(colormap(scaled_value))
                    texts[i][j][a].set_text(round(q_value[a], 2))
        return

    animation = FuncAnimation(fig, draw_func, frames=frames)
    animation.save(save_file)
    pbar.close()
    # plt.show()

if __name__ == "__main__":

    # read the arguments etc
    if len(sys.argv) < 2:
        print("usage : python analysis/process_data.py <list of json files")
        exit()

    json_handle = get_json_handle()

    # Only use the first handle for now?
    generatePlot(json_handle)

    exit()