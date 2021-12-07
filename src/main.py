import json
import numpy as np
import sys
import os, time
sys.path.append(os.getcwd())
import logging
from src.utils import globals, analysis_utils
from src.utils.formatting import create_file_name, pushup_metaParameters
from RlGlue import RlGlue
from src.experiment import ExperimentModel
from src.problems.registry import getProblem
from PyExpUtils.utils.Collector import Collector
from src.utils.rlglue import OneStepWrapper, OptionOneStepWrapper
from src.utils import rlglue
from src.utils.json_handling import get_sorted_dict, get_param_iterable
import copy
from src.data_management import zeo_common
from src.utils.run_utils import experiment_completed, InvalidRunException, save_error, cleanup_files, save_data
import argparse
import tqdm

# Automatically exits when it detects nan in Jax. No error handling yet though (probably need to add before sweep)
from jax.config import config
config.update("jax_debug_nans", True)

# Logging info level logs for visibility.
logging.basicConfig(level=logging.INFO)

t_start = time.time()

parser = argparse.ArgumentParser(description='Parallizable experiment run file')
parser.add_argument('json_path', help='path to the json that describes the configs to run')
parser.add_argument('idx', type = int, help='index of the json config for which to run')
parser.add_argument('-o', '--overwrite', action='store_true')
parser.add_argument('-p', '--progress', action='store_true', help='Show process')
parser.add_argument('-e', '--ignore-error', action='store_false', help='run the experiment even if it previously errored')
args = parser.parse_args()

json_file = args.json_path
idx = args.idx

exp = ExperimentModel.load(json_file)

max_steps = exp.max_steps
globals.collector = Collector()

# set random seeds accordingly
exp_json = exp.getPermutation(idx)
seed = exp_json['metaParameters']['seed']
np.random.seed(seed)

Problem = getProblem(exp.problem)
problem = Problem(exp, idx, seed)
agent = problem.getAgent()
env = problem.getEnvironment()

experiment_old_format = pushup_metaParameters(exp_json)

if args.overwrite == True:
    print('Will overwrite previous results when experiment finishes')

if experiment_completed(exp_json, args.ignore_error) and not args.overwrite:
    if (args.ignore_error):
        print('Counted run as completed if run errored previously')
    print(f'Run Already Complete - Ending Run')
    exit()

try:
    wrapper_class = agent.wrapper_class
    if (wrapper_class == rlglue.OptionFullExecuteWrapper or 
        wrapper_class == rlglue.OptionOneStepWrapper or 
        wrapper_class == rlglue.OneStepWrapper):
        wrapper = wrapper_class(agent, problem)
    else:
        raise NotImplementedError(f"wrapper class {wrapper_class} has not been implemented")

except AttributeError:
    print("main.py: Agent does not have a wrapper class stated, defaulting to parsing by strings")
    if "Option" in agent.__str__():
        wrapper = OptionOneStepWrapper(agent, problem)
    else:
        wrapper = OneStepWrapper(agent, problem)


save_logger_keys = ['Q', 'slow_Q', 'policy_agreement', 'reward']
print(f'Saved logger keys: {save_logger_keys}')

glue = RlGlue(wrapper, env)
# print("run:",run)
# Run the experiment
rewards = []
try:
    if exp.episodes > 0:
        episode_iter = range(exp.episodes)
        if args.progress:
            episode_iter = tqdm.tqdm(episode_iter)
        for episode in episode_iter:
            glue.total_reward = 0
            glue.runEpisode(max_steps)
            globals.collector.collect('return', glue.total_reward)
        globals.collector.reset()
    elif exp.episodes == -1:
        globals.blackboard['step_logging_interval'] = 500
        print('Running with steps rather than episodes')
        if (exp.max_steps == 0):
            raise ValueError('Running with step limit but max_steps is 0')
        
        step_iter = range(exp.max_steps)
        if args.progress:
            step_iter = tqdm.tqdm(step_iter)
        
        is_terminal = True
        for step in step_iter:
            if is_terminal:
                globals.collector.collect('return', glue.total_reward)
                is_terminal = False
                glue.total_reward = 0
                glue.start()
            _, _, _, is_terminal = glue.step()
        globals.collector.reset()
    else:
        raise NotImplementedError(f'Running {exp.episodes} episodes is not supported. Please either run with > 0 for fixed number of episodes or -1 to limit by step count instead')
except InvalidRunException as e:
    save_error(experiment_old_format, e)
    logging.info(f"Experiment errored {json_file} : {idx}, Time Taken : {time.time() - t_start}")
    exit(0)

save_obj = {}

# I'm pretty sure we need to subsample this especially once we have a larger dataset.
# [2021-12-03 chunlok] this subsampling should probably be done in the logging step rather than afterwards
for k in save_logger_keys:
    save_obj[k] = globals.collector.all_data[k]

# We likely want to abstract this away from src/main
cleanup_files(experiment_old_format)
save_data(experiment_old_format, save_obj)
logging.info(f"Experiment Done {json_file} : {idx}, Time Taken : {time.time() - t_start}")