import numpy as np
from typing import Callable, Tuple
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from datetime import datetime
import os

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

def scale_value(value: float, min_val:float, max_val:float, post_process_func: Callable= np.cbrt):
    percentage = (value - min_val) / (max_val - min_val)
    percentage = post_process_func(percentage)
    return percentage

def prompt_user_for_file_name(folder: str, prefix: str, suffix: str, extension: str, timestamp: bool=False):
    if timestamp:
        prompt_file_string = f"{prefix}<name>{suffix}--<timestamp>.{extension}"
    else:
        prompt_file_string = f"{prefix}<name>{suffix}.{extension}"

    experiment_name = input(f"Give the input for experiment name. File name will be '{prompt_file_string}': ")

    while (len(experiment_name) == 0):
        experiment_name = input("Please enter an experiment name that is longer than 0 length: ")
    
    # Since we'll be saving this, we're also making sure that the directory exists.
    os.makedirs(folder, exist_ok=True)

    if timestamp:
        time_str = datetime.now().strftime("%m-%d-%Y--%H-%M-%S")
        return f'{folder}/{prefix}{experiment_name}{suffix}--{time_str}.{extension}'
    else:
        return f'{folder}/{prefix}{experiment_name}{suffix}.{extension}'

def _plot_init(ax, columns: int, rows: int, center_arrows: bool = False):
    ax.set_xlim(0, columns)
    ax.set_ylim(0, rows)
    ax.invert_yaxis()

    texts = []
    patches = []
    arrows = []
    for r in range(rows):
        texts.append([])
        patches.append([])
        arrows.append([])
        for c in range(columns):
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


def prompt_episode_display_range(default_start_episode: int, defualt_end_episode: int, default_interval: int) -> Tuple[int, int, int]:
    start_episode = input(f"Enter in START episode (press enter for the default of {default_start_episode}): ")
    if len(start_episode) == 0:
        start_episode = default_start_episode
    else:
        start_episode = int(start_episode)

    end_episode = input(f"Enter in END episode (press enter for the default of {defualt_end_episode}): ")
    if len(end_episode) == 0:
        end_episode = defualt_end_episode
    else:
        end_episode = int(end_episode)

    interval = input(f"Enter in interval (press enter for the default of {default_interval}): ")
    if len(interval) == 0:
        interval = default_interval
    else:
        interval = int(interval)

    return start_episode, end_episode, interval