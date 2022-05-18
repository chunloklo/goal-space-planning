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

from tqdm import tqdm

# Mass taking imports from process_data for now
import os, sys, time, copy
sys.path.append(os.getcwd())
import numpy as np

from src.utils.json_handling import get_sorted_dict , get_param_iterable_runs
from src.utils.formatting import create_file_name, create_folder
from src.utils import analysis_utils
from src.analysis.gridworld_utils import _get_corner_loc, _get_q_value_patch_paths, get_file_name, get_text_location, prompt_user_for_file_name, get_action_offset, scale_value, _plot_init, prompt_episode_display_range
from src.analysis.plot_utils import get_x_range
from src.environments.HMaze import HMaze

from experiment_utils.sweep_configs.common import get_configuration_list_from_file_path
from src.analysis.plot_utils import load_configuration_list_data
from analysis.common import get_best_grouped_param, load_reward_rate, load_max_reward_rate
from  experiment_utils.analysis_common.configs import group_configs
from src.utils import run_utils
from analysis.common import load_data
import matplotlib
from datetime import datetime
from src.problems.PinballProblem import PinballHardProblem, PinballOracleProblem, PinballProblem, PinballSuboptimalProblem
from src.problems.registry import getProblem
from src.utils.log_utils import get_last_pinball_action_value_map

from experiment_utils.data_io.configs import load_all_db_configs_and_keys, load_data_from_config_id_zodb, save_data_zodb
from src.experiment import ExperimentModel
import pickle
import cloudpickle
import copy
if __name__ == "__main__":
    db_folder = './results_dbs/pinball_refactor_test'
    configs =  load_all_db_configs_and_keys(db_folder)
    print(len(configs))
    i = 0
    for config in configs:
        
        if config['load_model_name'] == 'pinball_hard_ddqn_value_policy_8_goals' and 'load_pretrain_goal_values' in config:
            data = load_data_from_config_id_zodb(db_folder, config.get_id())

            goal_values_name = config['load_pretrain_goal_values']

            new_config = copy.deepcopy(config)
            

            del new_config['load_pretrain_goal_values']

            new_config['load_goal_values_name'] = goal_values_name



            i +=  1

    # print(i)
    #         # get_data_co
            save_data_zodb(new_config, data)

            # print(i)
            # sdfsd
    exit()  