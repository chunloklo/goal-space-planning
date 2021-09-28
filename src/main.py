import json
import numpy as np
import sys
import os, time
sys.path.append(os.getcwd())
import logging
from src.utils import globals, analysis_utils
from src.utils.formatting import create_file_name
from RlGlue import RlGlue
from src.experiment import ExperimentModel
from src.problems.registry import getProblem
from PyExpUtils.utils.Collector import Collector
from src.utils.rlglue import OneStepWrapper, OptionOneStepWrapper
from src.utils import rlglue
from src.utils.json_handling import get_sorted_dict, get_param_iterable
import copy

# Logging info level logs for visibility.
logging.basicConfig(level=logging.INFO)

t_start = time.time()

if len(sys.argv) < 3:
    print('run again with:')
    print('python3 src/main.py <path/to/description.json> <idx>')
    exit(1)

# new stuff for parallel
#runs = sys.argv[1]
json_file = sys.argv[1]
idx = int(sys.argv[2])
# Get experiment
# d = get_sorted_dict(json_file)
# experiments = get_param_iterable(d)
# experiment = experiments[ idx % len(experiments)]
# seed = experiment['seed']

exp = ExperimentModel.load(json_file)

max_steps = exp.max_steps
globals.collector = Collector()
broke = False
# set random seeds accordingly

Problem = getProblem(exp.problem)
problem = Problem(exp, idx)
agent = problem.getAgent()
env = problem.getEnvironment()

experiment = copy.deepcopy(problem.params)
experiment['agent'] = exp.agent
experiment['problem'] = exp.problem
experiment['episodes'] = exp.episodes
seed = experiment['seed']
np.random.seed(seed)

#print("run:", seed)
#inner_idx = exp.numPermutations() * seed + idx

folder , filename = create_file_name(experiment)
if not os.path.exists(folder):
    time.sleep(2)
    try:
        os.makedirs(folder)
    except:
        pass


output_file_name = folder + filename
# Cut the run if already done
if os.path.exists(output_file_name + '.pkl'):
    print("Run Already Complete - Ending Run")
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

glue = RlGlue(wrapper, env)
# print("run:",run)
# Run the experiment
rewards = []
for episode in range(exp.episodes):
    glue.total_reward = 0
    glue.runEpisode(max_steps)
    if agent.FA()!="Tabular":
        # if the weights diverge to nan, just quit. This run doesn't matter to me anyways now.
        if np.isnan(np.sum(agent.w)):
            globals.collector.fillRest(np.nan, exp.episodes)
            broke = True
            break
    globals.collector.collect('return', glue.total_reward)
    globals.collector.collect('Q', np.copy(agent.Q))   
globals.collector.reset()

if broke:
    exit(0)


datum = globals.collector.all_data['return']
max_return = globals.collector.all_data['max_return']


analysis_utils.pkl_saver({
    'datum': datum,
    'max_return': max_return
}, output_file_name + '.pkl')



logging.info(f"Experiment Done {json_file} : {idx}, Time Taken : {time.time() - t_start}")