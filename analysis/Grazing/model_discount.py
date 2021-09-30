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
        save_file = save_folder + '/model_discount.mp4'

        if (not os.path.isdir(save_folder)):
            os.makedirs(save_folder)

        results = loadResults(exp, 'model_discount.npy')
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

        fig, axes = plt.subplots(1, figsize=(16, 16))
        ax = axes

        colormap = cm.get_cmap('viridis')

        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.invert_yaxis()
        texts, patches = _plot_init(ax)
    

        pbar = tqdm(total=data.shape[0])
        def draw_func(i):
            pbar.update(i - pbar.n)
            q_values = data[i, :, :]

            ax.set_title(f"episode: {i}")

            for i in range(10):
                for j in range(10):
                    q_value = q_values[i + j * 10, :]
                    # print(q_values.shape)
                    for a in range(3):
                        scaled_value = scale_value(q_value[a], min_val, max_val)
                        patches[i][j][a].set_facecolor(colormap(scaled_value))
                        texts[i][j][a].set_text(round(q_value[a], 2))
            return

        animation = FuncAnimation(fig, draw_func, frames=range(0, data.shape[0], 5))
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
