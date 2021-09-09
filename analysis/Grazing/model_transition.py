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

from action_values import _plot_init, scale_value

def generatePlot(ax, exp_paths, bounds):
    for exp_path in exp_paths:
        exp = ExperimentModel.load(exp_path)

        save_path = exp_path.replace('experiments', 'visualizations')
        save_folder = os.path.splitext(save_path)[0]
        save_file = save_folder + '/model_transition.mp4'

        if (not os.path.isdir(save_folder)):
            os.makedirs(save_folder)

        results = loadResults(exp, 'model_transition.npy')
        data = None
        for r in results:
            data = r.load()

        # Using a simple way of determining whether options are used.
        # Note that this might not work in the future if we do something separate, but it works for now
        # hasOptions = data.shape[-1] > 4
        print(f'data shape: {data.shape}')

        min_val = np.min(data)
        max_val = np.max(data)

        print(f'min: {min_val} max: {max_val}')

        fig, axes = plt.subplots(1, 3, figsize=(16 * 3, 16))
        ax = axes[0]

        colormap = cm.get_cmap('viridis')

        plot_texts = []
        plot_patches = []

        for axis in axes:
            texts, patches = _plot_init(axis)
            plot_texts.append(texts)
            plot_patches.append(patches)

        pbar = tqdm(total=data.shape[0])

        def draw_func(i):
            pbar.update(i - pbar.n)
            q_values = data[i, :, :]

            # titles.append(ax.set_title(f"frame: {i}"))
            ax.set_title(f"frame: {i}")

            for i in range(10):
                for j in range(10):
                    q_value = q_values[i + j * 10, :]
                    # print(q_values.shape)
                    termination_positions = [22, 27, 77, 100]
                    for o in range(3):
                        for p in range(4):
                            scaled_value = scale_value(q_value[o, termination_positions[p]], min_val, max_val)
                            plot_patches[o][i][j][p].set_facecolor(colormap(scaled_value))
                            plot_texts[o][i][j][p].set_text(round(q_value[o, termination_positions[p]], 2))
            return

        animation = FuncAnimation(fig, draw_func, frames=range(0, data.shape[0], 10))
        animation.save(save_file)
        pbar.close()
        # plt.show()

if __name__ == "__main__":
    # f, axes = plt.subplots(1)
    axes = None
    bounds = []

    exp_paths = sys.argv[1:]

    generatePlot(axes, exp_paths, bounds)

    # plt.show()
    exit()
