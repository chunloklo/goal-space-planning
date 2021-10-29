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
from src.utils.run_utils import experiment_completed
from src.data_management import zeo_common
from src.utils.run_utils import experiment_completed, InvalidRunException, save_error, cleanup_files
import argparse

# Logging info level logs for visibility.
logging.basicConfig(level=logging.INFO)

t_start = time.time()

parser = argparse.ArgumentParser(description='Parallizable experiment run file')
parser.add_argument('json_path', help='path to the json that describes the configs to run')
parser.add_argument('idx', type = int, help='index of the json config for which to run')
parser.add_argument('-o', '--overwrite', action='store_true')
parser.add_argument('-e', '--ignore-error', action='store_false', help='run the experiment even if it previously errored')
args = parser.parse_args()

json_file = args.json_path
idx = args.idx

exp = ExperimentModel.load(json_file)

max_steps = exp.max_steps
globals.collector = Collector()
broke = False
# set random seeds accordingly

Problem = getProblem(exp.problem)
problem = Problem(exp, idx)
agent = problem.getAgent()
env = problem.getEnvironment()

exp_json = exp.getPermutation(idx)
seed = exp_json['metaParameters']['seed']
np.random.seed(seed)
print(exp)

experiment_old_format = pushup_metaParameters(exp_json)
folder , filename = create_file_name(experiment_old_format)
output_file_name = folder + filename

if args.overwrite == True:
    print('Will overwrite previous results when experiment finishes')

if experiment_completed(exp_json, args.ignore_error) and not args.overwrite:
    if (args.ignore_error):
        print('Counted run as completed if run errored previously')
    print(f'Run Already Complete - Ending Run')
    exit()
else:
    if not os.path.exists(folder):
        time.sleep(2)
        try:
            os.makedirs(folder)
        except:
            pass
    
    cleanup_files(output_file_name)

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

glue = RlGlue(wrapper, env)
# print("run:",run)
# Run the experiment
rewards = []
try:
    for episode in range(exp.episodes):
        glue.total_reward = 0
        glue.runEpisode(max_steps)

        if globals.run_exception != None:
            break
        if agent.FA()!="Tabular":
            # if the weights diverge to nan, just quit. This run doesn't matter to me anyways now.
            if np.isnan(np.sum(agent.w)):
                globals.collector.fillRest(np.nan, exp.episodes)
                broke = True
                break
        globals.collector.collect('return', glue.total_reward)
    globals.collector.reset()
except InvalidRunException as e:
    print(f'Run errored out. Check {output_file_name}.err for info')
    save_error(output_file_name, e)
    logging.info(f"Experiment errored {json_file} : {idx}, Time Taken : {time.time() - t_start}")
    exit(0)

if broke:
    exit(0)

# I'm pretty sure we need to subsample this especially once we have a larger dataset.
datum = globals.collector.all_data['return']
max_return = globals.collector.all_data['max_return']

save_obj = {
    'datum': datum,
    'max_return': max_return,
    # Debug info. Comment me out if doing a big sweep I hope
    # 'Q': globals.collector.all_data['Q'],
    # 'model_r': globals.collector.all_data['model_r'],
    # 'model_discount': globals.collector.all_data['model_discount'],
    # 'model_transition': globals.collector.all_data['model_transition'],
    # 'end_goal': globals.collector.all_data['end_goal'],
    # 'goal_rewards': globals.collector.all_data['goal_rewards'],
    # 'action_selected': globals.collector.all_data['action_selected'],
}

# We likely want to abstract this away from src/main
if zeo_common.use_zodb():
    zeo_common.zodb_saver(save_obj, zeo_common.get_db_key(experiment_old_format))
else:
    analysis_utils.pkl_saver(save_obj, output_file_name + '.pkl')

logging.info(f"Experiment Done {json_file} : {idx}, Time Taken : {time.time() - t_start}")