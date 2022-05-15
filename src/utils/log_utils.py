from typing import Callable
from src.utils import globals
import numpy as np
from tqdm import tqdm

def run_if_should_log(log_func: Callable):
    try:
        if globals.blackboard['num_steps_passed'] % globals.blackboard['step_logging_interval'] == 0:
            log_func()
    except KeyError as e:
        pass

def get_last_pinball_action_value_map(num_outputs, get_q_func: Callable, resolution=40, show_progress=False):
    # Calculating the value at each state approximately
    NUM_ACTIONS = 5
    q_map = np.zeros((num_outputs, resolution, resolution, NUM_ACTIONS))

    y_iter =  enumerate(np.linspace(0, 1, resolution))
    if show_progress: y_iter = tqdm(y_iter, leave=None)

    for r, y in y_iter:

        x_iter =  enumerate(np.linspace(0, 1, resolution))
        if show_progress: x_iter = tqdm(x_iter, leave=None)

        for c, x in x_iter:
            q_map[:, r, c] = get_q_func(np.array([x, y, 0.0, 0.0]))
    return q_map