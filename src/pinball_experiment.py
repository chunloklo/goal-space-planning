import json
import numpy as np
import sys
import os, time
sys.path.append(os.getcwd())
import logging
from RlGlue import RlGlue
from src.experiment import ExperimentModel
from src.problems.registry import getProblem
from PyExpUtils.utils.Collector import Collector, FilteredCollector
from src.utils.rlglue import OneStepWrapper, OptionOneStepWrapper
from src.utils import rlglue
from src.utils.run_utils import InvalidRunException, save_error
import tqdm
from src.utils import globals
from experiment_utils.data_io.configs import save_data_zodb
from src.utils.param_utils import parse_param
import cloudpickle


def run(param: dict, aux_config={}):

    # Resetting blackboard every time this function is ran
    globals.blackboard = {}

    # # Don't import jax here if we don't need to
    # if aux_config.get('use_jax', False):
    import jax
    if aux_config.get('jax_debug_nans', False):
        # Automatically exits when it detects nan in Jax. No error handling yet though (probably need to add before sweep)
        jax.config.update("jax_debug_nans", True)

    jax.config.update('jax_platform_name', 'cpu')

    show_progress = aux_config.get('show_progress', False)

    t_start = time.time()

    # Logging info level logs for visibility.
    level = logging.CRITICAL
    logging.basicConfig(level=level)

    # Reading params
    max_steps = parse_param(param, 'max_steps', lambda p: p >= 0, default=0, optional=True)
    episodes = parse_param(param, 'episodes', lambda p: p >= 0, default=0, optional=True)
    seed = parse_param(param, 'seed', lambda p: isinstance(p, int), default=-1, optional=True)

    globals.param = param
    globals.aux_config = aux_config

    np.random.seed(seed)

    # [chunlok 2022-1-10] Massaging pararameter dict into the old experiment model parameters format.
    # This is clunky, but needed since we're dependent on the ExperimentModel for much of our initialization,
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
    # save_logger_keys = ['action_model_r', 'action_model_discount', 'action_model_transition', 'model_r', 'model_discount', 'model_transition', 'Q']
    save_logger_keys = ['Q', 'max_reward_rate', 'reward_rate']

    # Overriding logger keys from aux config if it exists:
    if 'log_keys' in param:
        save_logger_keys = param['log_keys']
        
    globals.collector = FilteredCollector(save_logger_keys)

    step_logging_interval = param['step_logging_interval']

    if show_progress:
        print(f'Saved logger keys: {save_logger_keys}')
        if exp.episodes == -1:
            print(f'Logging interval: {step_logging_interval}, num steps: {exp.max_steps}')
        input("Confirm run?")

    glue = RlGlue(wrapper, env)

    # Run the experiment
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
        elif exp.episodes <= 0:
            globals.blackboard['step_logging_interval'] = step_logging_interval
            logging.debug('Running with steps rather than episodes')
            if (exp.max_steps == 0):
                raise ValueError('Running with step limit but max_steps is 0')
            
            step_iter = range(exp.max_steps)
            if show_progress:
                step_iter = tqdm.tqdm(step_iter)
            
            is_terminal = True
            for _ in step_iter:
                if is_terminal:
                    globals.collector.collect('return', glue.total_reward)
                    is_terminal = False
                    glue.total_reward = 0
                    glue.start()
                _, _, _, is_terminal = glue.step()
            
            # Adding this here so that there's always something logged
            globals.collector.collect('num_steps_in_ep', 0)
            globals.collector.reset()
    except InvalidRunException as e:
        # [2022-01-24 chunlok] Leaving out error saving for now. This needs to be reimplemented in the data_io library still
        logging.critical(f"Experiment errored {param} : {idx}, Time Taken : {time.time() - t_start}")
        raise e

    save_obj = {}

    # # Calculating the value at each state approximately
    resolution = 40
    num_goals = problem.num_goals
    last_q_map = np.zeros((resolution, resolution, 5))
    last_goal_q_map = np.zeros((num_goals, resolution, resolution, 5))
    last_reward_map = np.zeros((num_goals, resolution, resolution, 5))
    last_gamma_map = np.zeros((num_goals, resolution, resolution, 5))
    for r, y in enumerate(np.linspace(0, 1, resolution)):
        for c, x in enumerate(np.linspace(0, 1, resolution)):
            last_q_map[r, c] = agent.behaviour_learner.get_action_values(np.array([x, y, 0.0, 0.0]))
            if param['agent'] == 'GSP_NN':
                for g in range(num_goals):
                    # goal_s = np.append([x, y, 0.0, 0.0], np.array(agent.goals[g]))
                    # action_value, reward, gamma = agent.goal_learner.get_goal_outputs(goal_s)
                    # SWITCHING OVER TO 1 NN PER GOAL
                    action_value, reward, gamma = agent.goal_learners[g].get_goal_outputs(np.array([x, y, 0.0, 0.0]))
                    last_goal_q_map[g, r, c] = action_value
                    last_reward_map[g, r, c] = reward
                    last_gamma_map[g, r, c] = gamma
                    pass

    # print(last_q_map.shape)
    globals.collector.collect('q_map', last_q_map)
    globals.collector.collect('goal_q_map', last_goal_q_map)
    globals.collector.collect('goal_r_map', last_reward_map)
    globals.collector.collect('goal_gamma_map', last_gamma_map)
    # globals.collector.collect('num_steps_in_ep', 0)
    globals.collector.reset()

    for k in save_logger_keys:
        if k in globals.collector.all_data:
            save_obj[k] = globals.collector.all_data[k]
        else:
            print(f'LOGGER KEY {k} WAS NOT SAVED BECAUSE IT WASNT FOUND IN THE COLLECTOR')
    
    save_data_zodb(param, save_obj)

    # Saving agent for display:

    # Saving the agent goal learners
    save_behavior = parse_param(param, 'save_behavior', lambda p: isinstance(p, bool), optional=True, default=False)
    if save_behavior:
        if param['agent'] == 'GSP_NN':
            cloudpickle.dump(agent, open('./src/environments/data/pinball/gsp_agent.pkl', 'wb'))
        else:
            cloudpickle.dump(agent, open('./src/environments/data/pinball/agent.pkl', 'wb'))
    # 'save_goal_model': [True],

    save_goal_learner = parse_param(param, 'save_state_to_goal_estimate', lambda p: isinstance(p, bool), optional=True, default=False)
    
    if save_goal_learner:
        # cloudpickle.dump(agent.goal_learner, open('./src/environments/data/pinball/goal_learner.pkl', 'wb'))
        # cloudpickle.dump(agent.goal_buffer, open('./src/environments/data/pinball/goal_buffer.pkl', 'wb'))
        
        # SWITCHING OVER TO HAVING ONE NN PER GOAL
        cloudpickle.dump(agent.goal_learners, open('./src/environments/data/pinball/goal_learner.pkl', 'wb'))
        cloudpickle.dump(agent.goal_buffers, open('./src/environments/data/pinball/goal_buffer.pkl', 'wb'))

    # if save_buffers:

    # cloudpickle.dump(agent.goal_estimate_learner, open('./src/environments/data/pinball/goal_estimate_learner.pkl', 'wb'))
    
    logging.info(f"Experiment Done {param} : {idx}, Time Taken : {time.time() - t_start}")
    return