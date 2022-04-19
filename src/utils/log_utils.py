from typing import Callable
from src.utils import globals
import numpy as np

def run_if_should_log(log_func: Callable):
    try:
        if globals.blackboard['num_steps_passed'] % globals.blackboard['step_logging_interval'] == 0:
            log_func()
    except KeyError as e:
        pass

def get_last_pinball_action_value_map(num_outputs, get_q_func: Callable):
    # # Calculating the value at each state approximately
    RESOLUTION = 40
    NUM_ACTIONS = 5
    q_map = np.zeros((num_outputs, RESOLUTION, RESOLUTION, NUM_ACTIONS))
    for r, y in enumerate(np.linspace(0, 1, RESOLUTION)):
        for c, x in enumerate(np.linspace(0, 1, RESOLUTION)):
            q_map[:, r, c] = get_q_func(np.array([x, y, 0.0, 0.0]))
    return q_map