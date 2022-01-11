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
from src.agents.registry import getAgent
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
from src.utils.param_utils import parse_param


def run(param: dict, show_progress: bool = False):
    config.update("jax_debug_nans", True)
    config.update('jax_platform_name', 'cpu')

    t_start = time.time()

    # Logging info level logs for visibility.
    logging.basicConfig(level=logging.INFO)

    # Reading params
    max_steps = parse_param(param, 'max_steps', lambda p: p >= 0, default=0, optional=True)
    episodes = parse_param(param, 'episodes', lambda p: p >= 0, default=0, optional=True)
    seed = parse_param(param, 'seed', lambda p: isinstance(p, int), default=-1, optional=True)

    globals.collector = Collector()

    np.random.seed(seed)

    # [chunlok 2022-1-10] Massaging pararameter dict into the old experiment model parameters format.
    # This is clunky, but needed since we're dependent on the ExperimentModel for much of our initiations,
    # which has a specific paradigm on what an experiment is.
    # TODO: Need to rip out the experiment description/model from much of the code. There's too much
    # assumption in that code to make things work well with other pieces of code.
    exp_params = {
        'agent': param['agent'],
        'problem': param['problem'],
        'max_steps': max_steps,
        'episodes': episodes,
        'metaParameters': param
    }
    # index will always be 0 since there's only 1 parameter there
    idx = 0

    exp = ExperimentModel.load_from_params(exp_params)
    max_steps = exp.max_steps

    # set random seeds accordingly
    exp_json = exp.getPermutation(idx)
    seed = exp_json['metaParameters']['seed']
    np.random.seed(seed)

    Problem = getProblem(exp.problem)
    problem = Problem(exp, idx, seed)
    agent = problem.getAgent()
    env = problem.getEnvironment()

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


    # save_logger_keys = ['tau', 'Q', 'max_reward_rate', 'reward_rate']
    save_logger_keys = ['action_model_r', 'action_model_discount', 'action_model_transition', 'model_r', 'model_discount', 'model_transition']
    # save_logger_keys = []

    step_logging_interval = 100

    if show_progress:
        print(f'Saved logger keys: {save_logger_keys}')
        if exp.episodes == -1:
            print(f'Logging interval: {step_logging_interval}, num steps: {exp.max_steps}')
        input("Confirm run?")

    glue = RlGlue(wrapper, env)

    # Run the experiment
    rewards = []
    try:
        if exp.episodes > 0:
            episode_iter = range(exp.episodes)
            if show_progress:
                episode_iter = tqdm.tqdm(episode_iter)
            for episode in episode_iter:
                glue.total_reward = 0
                glue.runEpisode(max_steps)
                globals.collector.collect('return', glue.total_reward)
            globals.collector.reset()
        elif exp.episodes == -1:
            globals.blackboard['step_logging_interval'] = step_logging_interval
            print('Running with steps rather than episodes')
            if (exp.max_steps == 0):
                raise ValueError('Running with step limit but max_steps is 0')
            
            step_iter = range(exp.max_steps)
            if show_progress:
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
        save_error(param, e)
        logging.info(f"Experiment errored {param} : {idx}, Time Taken : {time.time() - t_start}")
        return

    save_obj = {}

    # I'm pretty sure we need to subsample this especially once we have a larger dataset.
    # [2021-12-03 chunlok] this subsampling should probably be done in the logging step rather than afterwards
    for k in save_logger_keys:
        save_obj[k] = globals.collector.all_data[k]

    # We likely want to abstract this away from src/main
    cleanup_files(param)
    save_data(param, save_obj)
    logging.info(f"Experiment Done {param} : {idx}, Time Taken : {time.time() - t_start}")
    return