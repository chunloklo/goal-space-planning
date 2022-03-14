
from concurrent.futures import process
import numpy as np
from PyExpUtils.utils.random import argmax, choice
import random
from src.agents.components.learners import ESarsaLambda, QLearner, QLearner_ImageNN
from src.agents.components.search_control import ActionModelSearchControl_Tabular
from src.environments.GrazingWorldAdam import GrazingWorldAdamImageFeature, get_pretrained_option_model, state_index_to_coord
from src.utils import rlglue
from src.utils import globals
from src.utils import options, param_utils
from src.agents.components.models import OptionModel_TB_Tabular, OptionModel_Sutton_Tabular
from src.agents.components.approximators import DictModel
from PyFixedReps.BaseRepresentation import BaseRepresentation
import numpy.typing as npt
from PyFixedReps.Tabular import Tabular
from typing import Dict, List, Optional, Union, Tuple, Any, TYPE_CHECKING
import jax.numpy as jnp
from src.agents.components.buffers import Buffer, DictBuffer
from collections import namedtuple
from src.environments.GrazingWorldAdam import get_all_transitions
import jax
import haiku as hk
import optax
import copy
import pickle
import traceback


from src.utils.log_utils import run_if_should_log
from src.utils.numpy_utils import create_onehot

if TYPE_CHECKING:
    # Important for forward reference
    from src.problems.BaseProblem import BaseProblem

# [2022-03-09 chunlok] Copied over from DynaOptions_NN. For now, getting this to work with Pinball env.
class GSP_NN:
    def __init__(self, problem: 'BaseProblem'):
        self.wrapper_class = rlglue.OneStepWrapper
        self.env = problem.getEnvironment()
        self.num_actions = problem.actions
        self.params = problem.params
        self.random = np.random.RandomState(problem.seed)
        self.goals = problem.goals
        self.num_goals = len(self.goals)
        self.goal_termination_func = problem.goal_termination_func
        self.goal_initiation_func = problem.goal_initiation_func

        params = self.params
        self.step_size = 1e-3
        self.batch_size = 16
        self.epsilon = param_utils.parse_param(params, 'epsilon', lambda p : isinstance(p, float) and p >= 0.0)

        self.behaviour_learner = QLearner_NN((4,), 5, self.step_size)

        self.goal_learners: List[GoalLearner_NN] = []
        for _ in range(self.num_goals):
            # We have 25 NNs????
            self.goal_learners.append(GoalLearner_NN((4,), self.num_actions, self.step_size))

        self.buffer_size = 200000
        self.min_buffer_size_before_update = 100
        self.buffer = Buffer(self.buffer_size, {'x': (4,), 'a': (), 'xp': (4,), 'r': (), 'gamma': ()}, self.random.randint(0,2**31))

        self.goal_buffers = []
        for _ in range(self.num_goals):
            self.goal_buffers.append(Buffer(self.buffer_size, {'x': (4,), 'a': (), 'xp': (4,), 'r': (), 'gamma': (), 'goal_policy_cumulant': (), 'goal_discount': ()}, self.random.randint(0,2**31), type_map={'a': np.int64}))

        self.goal_estimate_learner = GoalEstimates(self.num_goals)
        self.num_steps_in_ep = 0

        self.goal_value_learner = GoalValueLearner(self.num_goals)


        # self.use_pretrained_behavior = param_utils.parse_param(params, 'use_pretrained_behavior', lambda p : isinstance(p, bool), optional=True, default=False)

        # if self.use_pretrained_behavior:
        #     behavior_weights = pickle.load(open('src/environments/data/pinball/behavior_params.pkl', 'rb'))
        #     self.behaviour_learner.params = behavior_weights
        #     self.behaviour_learner.target_params = copy.deepcopy(behavior_weights)

        self.use_pretrained_model = param_utils.parse_param(params, 'use_pretrained_model', lambda p : isinstance(p, bool), optional=True, default=False)
        if self.use_pretrained_model:
            self.goal_learners = pickle.load(open('./src/environments/data/pinball/goal_learners.pkl', 'rb'))
            self.goal_estimate_learner = pickle.load(open('./src/environments/data/pinball/goal_estimate_learner.pkl', 'rb'))

        self.cumulative_reward = 0

    def FA(self):
        return "Neural Network"

    def __str__(self):
        return "GSP_NN"

    def get_policy(self, s) -> npt.ArrayLike:
        probs = np.zeros(self.num_actions)
        probs += self.epsilon / (self.num_actions)
        action_values = self.behaviour_learner.get_action_values(s)

        a = np.argmax(action_values)

        probs[a] += 1 - self.epsilon
        return probs


    # public method for rlglue
    def selectAction(self, s: Any) -> int:
        s = np.array(s)
        a = self.random.choice(self.num_actions, p = self.get_policy(s))
        return a

    def update_behavior(self):
        if self.buffer.num_in_buffer < self.min_buffer_size_before_update:
            # Not enough samples in buffer to update yet
            return 
        # Updating behavior
        data = self.buffer.sample(self.batch_size)
        self.behaviour_learner.update(data, polyak_stepsize=0.001)

        for g in range(self.num_goals):
            if self.goal_buffers[g].num_in_buffer < self.min_buffer_size_before_update:
                # Not enough samples in buffer to update yet
                continue 
            data = self.goal_buffers[g].sample(self.batch_size)
            self.goal_learners[g].update(data, polyak_stepsize=0.0001)

    def update(self, s: Any, a, sp: Any, r, gamma, terminal: bool = False):
        # print(s)
        # if s == [0.2, 0.9, 0.0, 0.0]:
        #     for i, goal_learner in enumerate(self.goal_policy_learners):
        #         print(f' {i}: {goal_learner.get_action_values(np.array(s))}')

        # Calculating the value at each state approximately
        # term_map = np.zeros((20, 20))
        # for row, y in enumerate(np.linspace(0, 1, 20)):
        #     for c, x in enumerate(np.linspace(0, 1, 20)):
        #         goal_terms = self.goal_termination_func(np.array([x, y, 0, 0]))
        #         term_map[row, c] = 1 if goal_terms[5] == True else 0

        # init_map = np.zeros((20, 20))
        # for row, y in enumerate(np.linspace(0, 1, 20)):
        #     for c, x in enumerate(np.linspace(0, 1, 20)):
        #         goal_init = self.goal_initiation_func(np.array([x, y, 0, 0]))
        #         init_map[row, c] = 1 if goal_init[5] == True else 0

        # print('map')
        # print(term_map)
        # print(init_map)

        # if terminal:
            # print('terminated!')
        goal_terms = self.goal_termination_func(sp)

        goal_discount = np.copy(goal_terms)
        goal_discount[goal_terms == True] = 0
        goal_discount[goal_terms == False] = gamma

        # print(gamma)

        goal_policy_cumulant = np.copy(goal_terms)
        goal_policy_cumulant[goal_terms == True] = 1
        goal_policy_cumulant[goal_terms == False] = 0

        goal_init = self.goal_initiation_func(s)

        for g in range(self.num_goals):
            if goal_init[g]:
                self.goal_buffers[g].update({'x': s, 'a': a, 'xp': sp, 'r': r, 'gamma': gamma, 'goal_policy_cumulant': goal_policy_cumulant[g], 'goal_discount': goal_discount[g]})

        self.buffer.update({'x': s, 'a': a, 'xp': sp, 'r': r, 'gamma': gamma})
        self.num_steps_in_ep += 1


        self.update_behavior()

        goal_policies = np.zeros((self.num_goals, self.num_actions))
        goal_rs = np.zeros((self.num_goals, self.num_actions))
        goal_gammas = np.zeros((self.num_goals, self.num_actions))
        for g in range(self.num_goals):
            action_value, goal_r, goal_gamma = self.goal_learners[g].get_goal_outputs(np.array(s))
            goal_policies[g, np.argmax(action_value)] = 1 
            goal_rs[g] = goal_r
            goal_gammas[g] = goal_gamma
            pass
        self.goal_estimate_learner.update(s, goal_policies, goal_r, goal_gammas, 0.01, r, sp, goal_terms, self.behaviour_learner.get_action_values(np.array(s)), goal_init)
    
        self.goal_value_learner.update(self.goal_estimate_learner.gamma, self.goal_estimate_learner.r, 0, self.goal_estimate_learner.goal_init, self.goal_estimate_learner.goal_baseline)

        # # Logging
        self.cumulative_reward += r
        def log():
            globals.collector.collect('reward_rate', np.copy(self.cumulative_reward) / globals.blackboard['step_logging_interval'])
            self.cumulative_reward = 0
        run_if_should_log(log)

        if not terminal:
            ap = self.selectAction(sp)
        else:
            ap = None

        return ap

    def agent_end(self, s, a, r, gamma):
        self.update(s, a, s, r, 0, terminal=True)

        globals.collector.collect('num_steps_in_ep', self.num_steps_in_ep)
        self.num_steps_in_ep = 0
        # self.behaviour_learner.episode_end()
        # self.option_model.episode_end()

class GoalLearner_NN():
    def __init__(self, state_shape, num_actions: int, learning_rate: float):
        self.num_actions: int = num_actions

        self.num_outputs: int = num_actions * 3 # adding reward and gamma output
        self.policy_slice = slice(0, num_actions) # slice for the greedy goal reaching policy
        self.reward_slice = slice(num_actions, num_actions * 2) # slice for the reward following greedy policy till termination
        self.gamma_slice = slice(num_actions * 2, num_actions * 3)# slice for the discount following greedy policy till termination
        
        # Initializing jax functions
        def q_function(states):
            mlp = hk.Sequential([
                hk.Linear(256), jax.nn.relu,
                hk.Linear(256), jax.nn.relu,
                hk.Linear(self.num_outputs),
            ])
            return mlp(states) 
        
        self.f_qfunc = hk.without_apply_rng(hk.transform(q_function))
        self.f_opt = optax.adam(learning_rate)

        def _take_action_index(data, action):
            return jnp.take_along_axis(data, jnp.expand_dims(action, axis=1), axis=1).squeeze()
    
        def _loss(params: hk.Params, target_params: hk.Params, data):
            r = data['r']
            x = data['x']
            a = data['a']
            xp = data['xp']
            gamma = data['gamma']
            goal_policy_cumulant = data['goal_policy_cumulant']
            goal_discount = data['goal_discount']

            x_pred = self.f_qfunc.apply(params, x)
            policy_pred = _take_action_index(x_pred[:, self.policy_slice], a)
            reward_pred = _take_action_index(x_pred[:, self.reward_slice], a)
            gamma_pred = _take_action_index(x_pred[:, self.gamma_slice], a)

            xp_pred = jax.lax.stop_gradient(self.f_qfunc.apply(target_params, xp))
            # Using the target params here for selecting the next action to be more stable
            xp_ap = jnp.argmax(xp_pred[:, self.policy_slice], axis=1)
            # print(xp_ap.shape)
            # print(goal_discount.shape)
            # print(goal_policy_cumulant.shape)
            # print()
            policy_target = goal_policy_cumulant + goal_discount * _take_action_index(xp_pred[:, self.policy_slice], xp_ap)
            reward_target = r + goal_discount * _take_action_index(xp_pred[:, self.reward_slice], xp_ap)
            # print(_take_action_index(xp_pred[:, self.reward_slice], xp_ap).shape)

            gamma_cumulant = jnp.where(goal_discount == 0, gamma, 0)
            gamma_target = gamma_cumulant + goal_discount *  _take_action_index(xp_pred[:, self.gamma_slice], xp_ap)

            # print(jnp.mean(jnp.square((policy_target-policy_pred))).shape)

            policy_loss = jnp.mean(jnp.square(policy_target - policy_pred))
            reward_loss = jnp.mean(jnp.square(reward_target - reward_pred))
            gamma_loss = jnp.mean(jnp.square(gamma_target - gamma_pred))
            
            # return policy_loss
            return policy_loss + reward_loss + gamma_loss


        def _get_goal_output(params: hk.Params, x: Any):
            output = self.f_qfunc.apply(params, x)
            return output[..., self.policy_slice], output[..., self.reward_slice], output[..., self.gamma_slice]

        def _update(params: hk.Params, target_params: hk.Params, opt_state, data):
            grads = jax.grad(_loss)(params, target_params, data)
            updates, opt_state = self.f_opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state
            
        self.f_get_goal_output = jax.jit(_get_goal_output)
        self.f_update = jax.jit(_update)
        # self.f_update = _update

        # Initialization
        dummy_state = jnp.zeros(state_shape)
        self.params = self.f_qfunc.init(jax.random.PRNGKey(42), dummy_state)
        self.opt_state = self.f_opt.init(self.params)

        # target params for the network
        self.target_params = copy.deepcopy(self.params)

    # def get_multi_goal_outputs(self, x: npt.ArrayLike) -> np.ndarray:
    #     return self.f_get_goal_output(self.params, x)

    def get_goal_outputs(self, x: npt.ArrayLike) -> np.ndarray:
        return self.f_get_goal_output(self.params, x)

    def update(self, data, polyak_stepsize:float=0.005):
        self.params, self.opt_state = self.f_update(self.params, self.target_params, self.opt_state, data)
        self.target_params = optax.incremental_update(self.params, self.target_params, polyak_stepsize)
        # return self.params

class QLearner_funcs():
    def __init__(self, num_actions: int, learning_rate: float):
        self.num_actions = num_actions

        def q_function(states):
            mlp = hk.Sequential([
                hk.Linear(128), jax.nn.relu,
                hk.Linear(128), jax.nn.relu,
                hk.Linear(self.num_actions),
            ])
            return mlp(states) 
        self.f_qfunc = hk.without_apply_rng(hk.transform(q_function))
        self.f_opt = optax.adam(learning_rate)

        def get_q_values(params: hk.Params, x: Any):
            action_values = self.f_qfunc.apply(params, x)
            return action_values
        
        def get_td_errors(params: hk.Params, target_params: hk.Params, data):
            r = data['r']
            x = data['x']
            a = data['a']
            xp = data['xp']
            gamma = data['gamma']

            x_pred = self.f_qfunc.apply(params, x)
            xp_pred = jax.lax.stop_gradient(jnp.max(self.f_qfunc.apply(target_params, xp), axis=1))
            prev_pred = jnp.take_along_axis(x_pred, jnp.expand_dims(a, axis=1), axis=1).squeeze()
            td_error = r + gamma * xp_pred - prev_pred
            return td_error

        def esarsa_loss(params: hk.Params, target_params: hk.Params, data):
            r = data['r']
            x = data['x']
            a = data['a']
            xp = data['xp']
            gamma = data['gamma']
            xp_policy = data['xp_policy']

            x_pred = self.f_qfunc.apply(params, x)
            print((self.f_qfunc.apply(target_params, xp) * xp_policy).shape)
            xp_pred = jax.lax.stop_gradient(jnp.sum(self.f_qfunc.apply(target_params, xp) * xp_policy))
            prev_pred = jnp.take_along_axis(x_pred, jnp.expand_dims(a, axis=1), axis=1).squeeze()
            td_error = r + gamma * xp_pred - prev_pred
            return jnp.mean(jnp.square(td_error)), td_error
            
        def loss(params: hk.Params, target_params: hk.Params, data):
            td_errors = get_td_errors(params, target_params, data)
            return  jnp.mean(jnp.square(td_errors)), td_errors

        def update(params: hk.Params, target_params: hk.Params, opt_state, data):
            grads, td_errors = jax.grad(loss, has_aux=True)(params, target_params, data)
            updates, opt_state = self.f_opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, td_errors

        def esarsa_update(params: hk.Params, target_params: hk.Params, opt_state, data):
            grads, td_errors = jax.grad(esarsa_loss, has_aux=True)(params, target_params, data)
            updates, opt_state = self.f_opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, td_errors

        self.f_get_q_values = jax.jit(get_q_values)
        self.f_update = jax.jit(update)  
        self.f_get_td_errors = jax.jit(get_td_errors)
        self.f_esarsa_update = jax.jit(esarsa_update)      
        return

class QLearner_NN():
    def __init__(self, state_shape, num_actions: int, learning_rate: float):
        self.num_actions: int = num_actions
        self.funcs = QLearner_funcs(num_actions, learning_rate)

        dummy_state = jnp.zeros(state_shape)
        self.params = self.funcs.f_qfunc.init(jax.random.PRNGKey(42), dummy_state)
        self.opt_state = self.funcs.f_opt.init(self.params)

        # target params for the network
        self.target_params = copy.deepcopy(self.params)

    def get_action_values(self, x: npt.ArrayLike) -> np.ndarray:
        action_values = self.funcs.f_get_q_values(self.params, x)
        return action_values

    def get_target_action_values(self, x: npt.ArrayLike) -> np.ndarray:
        action_values = self.funcs.f_get_q_values(self.target_params, x)
        return action_values
    
    def get_td_errors(self, data):
        return self.funcs.f_get_td_errors(self.params, self.target_params, data)

    def update(self, data, polyak_stepsize:float=0.005):
        self.params, self.opt_state, td_errors = self.funcs.f_update(self.params, self.target_params, self.opt_state, data)
        self.target_params = optax.incremental_update(self.params, self.target_params, polyak_stepsize)
        return self.params, td_errors

    def esarsa_update(self, data, polyak_stepsize:float=0.005):
        self.params, self.opt_state, td_errors = self.funcs.f_esarsa_update(self.params, self.target_params, self.opt_state, data)
        self.target_params = optax.incremental_update(self.params, self.target_params, polyak_stepsize)
        return self.params, td_errors


class GoalEstimates:
    def __init__(self, num_goals):
        self.num_goals = num_goals

        # initializing weights
        self.r = np.zeros((self.num_goals, self.num_goals))
        self.gamma = np.zeros((self.num_goals, self.num_goals))
        self.goal_baseline = np.zeros(self.num_goals)
        self.goal_init = np.zeros((self.num_goals, self.num_goals))

    def update(self, x, option_pi_x, r_s, gamma_s, alpha, r, xp,  on_goal, x_action_values, goal_init):
        for g in range(self.num_goals):
            if on_goal[g] == True: 
                # if g == 12:
                #     print(goal_init)
                self.r[g] += alpha * (np.sum(option_pi_x * r_s, axis=1) - self.r[g])
                self.gamma[g] += alpha * (np.sum(option_pi_x * gamma_s, axis=1)- self.gamma[g])
                self.goal_baseline[g] += alpha * (np.argmax(x_action_values))
                self.goal_init[g] += alpha * 10 * (goal_init - self.goal_init[g])

        def log():
            globals.collector.collect('goal_r', np.copy(self.r))
            globals.collector.collect('goal_gamma', np.copy(self.gamma))
            globals.collector.collect('goal_baseline', np.copy(self.goal_baseline))
            globals.collector.collect('goal_init', np.copy(self.goal_init))
        run_if_should_log(log)

class GoalValueLearner:
    def __init__(self, num_goals):
        self.num_goals = num_goals
        
        # Initializing goal values
        self.goal_values = np.zeros(self.num_goals)
    
    def update(self, goal_gamma, reward_goals, goal_bonus, goal_init, goal_baseline):
        num_planning_steps = 1

        goal_gamma = np.clip(goal_gamma, 0, 1)

        for _ in range(num_planning_steps):
            # Just doing value iteration for now 

            # Can we possibly vectorize this?
            for g in range(self.num_goals):
                # print(goal_gamma[g])
                returns = reward_goals[g] + goal_gamma[g] * (self.goal_values + goal_bonus)

                valid_goals = np.nonzero(goal_init[g])
                # print(goal_gamma[g, valid_goals])

                if len(valid_goals[0]) > 0:
                    self.goal_values[g] = max(goal_baseline[g], np.max(returns[valid_goals[0]]))
        
        if globals.blackboard['num_steps_passed'] % globals.blackboard['step_logging_interval'] == 0:
            globals.collector.collect('goal_values', np.copy(self.goal_values)) 