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
    data = data["model_transition"]
    print(data.shape)
    data = data[:, 0, :, :, :]
    data = np.mean(data, axis=0)
    # asdasd
    # data = load_experiment_data(exp_path, file_name)
    experiment_name = get_experiment_name()

    anim_file_name = f'{experiment_name}_model_transition.mp4'
    save_path = "./visualizations/"
    save_folder = os.path.splitext(save_path)[0]
    save_file = save_folder + f'/{anim_file_name}'

    if (not os.path.isdir(save_folder)):
        os.makedirs(save_folder)


    min_val = np.min(data)
    max_val = np.max(data)
    print(f'min: {min_val} max: {max_val}')

    colormap = cm.get_cmap('viridis')

    num_options = 3

    fig, axes = plt.subplots(1, num_options, figsize=(16 * num_options, 16))
    ax = axes[0]
    
    plot_texts = []
    plot_patches = []

    for axis in axes:
        texts, patches = _plot_init(axis)
        plot_texts.append(texts)
        plot_patches.append(patches)

    
    print(data.shape)

    # max_frames = 50
    # interval = 1
    max_frames = data.shape[0]
    interval = 5
    frames = range(0, max_frames, interval)
    pbar = tqdm(total=max_frames)

    def draw_func(i):
        pbar.update(i - pbar.n)
        q_values = data[i, :, :, :]

        ax.set_title(f"episode: {i}")

        for i in range(10):
            for j in range(10):
                q_value = q_values[i + j * 10, :]
                # print(q_values.shape)
                termination_positions = [22,27,77,100]
                for o in range(num_options):
                    for p in range(len(termination_positions)):
                        scaled_value = scale_value(q_value[o, termination_positions[p]], min_val, max_val)
                        plot_patches[o][i][j][p].set_facecolor(colormap(scaled_value))
                        plot_texts[o][i][j][p].set_text(round(q_value[o, termination_positions[p]], 2))
        return

    animation = FuncAnimation(fig, draw_func, frames=frames)
    animation.save(save_file)
    pbar.close()

if __name__ == "__main__":

    # read the arguments etc
    if len(sys.argv) < 2:
        print("usage : python analysis/process_data.py <list of json files")
        exit()


    json_handle = get_json_handle()

    # Only use the first handle for now?
    generatePlot(json_handle)

    exit()