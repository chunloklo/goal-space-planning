
from concurrent.futures import process
from re import L
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

        self.behaviour_learner = QLearner_NN((4,), 5, self.step_size, self.epsilon)

        self.goal_features = 2
        self.goal_learner = GoalLearner_NN((4 + self.goal_features,), self.num_actions, self.step_size)
        # self.goal_learners: List[GoalLearner_NN] = []
        
        # for _ in range(self.num_goals):
        #     # We have 25 NNs????
        #     self.goal_learners.append(GoalLearner_NN((4,), self.num_actions, self.step_size))

        self.buffer_size = 1000000
        self.min_buffer_size_before_update = 1000
        self.buffer = Buffer(self.buffer_size, {'x': (4,), 'a': (), 'xp': (4,), 'r': (), 'gamma': ()}, self.random.randint(0,2**31))

        self.goal_buffer = Buffer(self.buffer_size * self.num_goals, {'x': (4 + 2,), 'a': (), 'xp': (4 + 2,), 'r': (), 'gamma': (), 'goal_policy_cumulant': (), 'goal_discount': ()}, self.random.randint(0,2**31), type_map={'a': np.int64})
        self.goal_estimate_learner = GoalEstimates(self.num_goals)
        self.num_steps_in_ep = 0

        self.OCI_update_interval = param_utils.parse_param(params, 'OCI_update_interval', lambda p : isinstance(p, int) and p >= 0) # Number of update steps between each OCI update

        self.goal_value_learner = GoalValueLearner(self.num_goals)
    

        self.use_pretrained_behavior = param_utils.parse_param(params, 'use_pretrained_behavior', lambda p : isinstance(p, bool), optional=True, default=False)

        if self.use_pretrained_behavior:
            self.behaviour_learner = pickle.load(open('src/environments/data/pinball/behavior_learner.pkl', 'rb'))
            print('using pretrained behavior')

        self.use_pretrained_model = param_utils.parse_param(params, 'use_pretrained_model', lambda p : isinstance(p, bool), optional=True, default=False)
        if self.use_pretrained_model:
            self.goal_learner = pickle.load(open('./src/environments/data/pinball/goal_learner.pkl', 'rb'))
            
        self.cumulative_reward = 0
        self.num_term = 0
        self.num_updates = 0

        self.tau = np.zeros(self.num_goals)

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
    
    def _add_to_buffer(self, s, a, sp, r, gamma, terminal, goal_init, goal_terms):
        self.buffer.update({'x': s, 'a': a, 'xp': sp, 'r': r, 'gamma': gamma})
        
        sp_goal_init = self.goal_initiation_func(sp)

        goal_discount = np.empty(goal_terms.shape)
        goal_discount[goal_terms == True] = 0
        goal_discount[goal_terms == False] = gamma

        goal_policy_cumulant = np.empty(goal_terms.shape)
        goal_policy_cumulant[goal_terms == True] = gamma
        goal_policy_cumulant[goal_terms == False] = 0

        for g in range(self.num_goals):
            # Don't learn with experience that takes you "outside" of the goal initiation zone.
            if goal_init[g] and (goal_terms[g] or sp_goal_init[g] != False):
                goal_s = np.append(s, np.array(self.goals[g]))
                goal_sp = np.append(sp, np.array(self.goals[g]))
                self.goal_buffer.update({'x': goal_s, 'a': a, 'xp': goal_sp, 'r': r, 'gamma': gamma, 'goal_policy_cumulant': goal_policy_cumulant[g], 'goal_discount': goal_discount[g]})

    def _direct_rl_update(self):
        if self.buffer.num_in_buffer >= self.min_buffer_size_before_update:
            # Updating behavior
            data = self.buffer.sample(self.batch_size)
            self.behaviour_learner.update(data, polyak_stepsize=0.0001)

    def _state_to_goal_estimate_update(self):
        if self.goal_buffer.num_in_buffer >= self.min_buffer_size_before_update * self.num_goals:
            num_batches = 1
            for _ in range(num_batches):
                data = self.goal_buffer.sample(32)
                self.goal_learner.update(data, polyak_stepsize=0.0001)

    def _goal_estimate_update(self, s, a, sp, r, gamma, terminal, goal_init, goal_terms):
        # Num features = 4
        goal_s = np.empty((self.num_goals, 6))
        for g in range(self.num_goals):
            goal_s[g] = np.append(s, np.array(self.goals[g]))

        action_value, goal_r, goal_gammas = self.goal_learner.get_goal_outputs(goal_s)

        goal_policies = np.zeros((self.num_goals, self.num_actions))
        goal_policies[:, np.argmax(action_value, axis=1)] = 1 
        
        current_val =  self.behaviour_learner.get_action_values(np.array(s))
        self.goal_estimate_learner.update(s, goal_policies, goal_r, goal_gammas, 0.01, r, sp, goal_terms, current_val, goal_init)
        # self.goal_estimate_learner.update(None, None, None, None, 0.1, None, None, goal_terms, current_val, None)

    def _goal_value_update(self):
        # bonus = np.where(self.tau > 12000, 1000, 0)
        bonus = 0
        goal_gamma = np.clip(self.goal_estimate_learner.gamma, 0, 1)
        self.goal_value_learner.update(goal_gamma, self.goal_estimate_learner.r, bonus, self.goal_estimate_learner.goal_init, self.goal_estimate_learner.goal_baseline)

    # def _OCI_combined(self):
    #     ############################ OCI
    #     if self.buffer.num_in_buffer >= self.min_buffer_size_before_update:
    #         # OCI. Resampling here
    #         OCI_batch_size = self.batch_size
    #         data = self.buffer.sample(OCI_batch_size)
    #         # data['x'] = np.vstack(data['x']
    #         x_goals_list = []

    #         goals = np.array(self.goals)
    #         for i in range(OCI_batch_size):
    #             x_goal_init = self.goal_initiation_func(data['x'][i])
    #             x_valid_goals = goals[np.where(x_goal_init == True)]

    #             tiled_x = np.tile(data['x'][i], (x_valid_goals.shape[0], 1))
    #             goal_x = np.append(tiled_x, x_valid_goals, axis=1)

    #             x_goals_list.append(goal_x)
            
    #         all_goal_x = np.vstack(x_goals_list)
    #         _, goal_rewards, goal_discounts = self.goal_learner.get_goal_outputs(all_goal_x)
            
    #         goal_targets = []
    #         start_index = 0
    #         # bonus = np.where(self.tau > 12000, 1000, 0)
    #         bonus = 0
    #         # goal_value_with_bonus = self.goal_value_learner.goal_values + bonus
    #         goal_value_with_bonus = np.copy(self.goal_estimate_learner.goal_baseline)

    #         action_values = self.behaviour_learner.get_action_values(data['x'])
            
    #         for i in range(OCI_batch_size):
    #             x_num_goals = x_goals_list[i].shape[0]
    #             end_index = start_index + x_num_goals
    #             x_goal_init = self.goal_initiation_func(data['x'][i])
                
    #             x_goal_reward = goal_rewards[start_index: end_index]
    #             x_goal_discount = np.clip(goal_discounts[start_index: end_index], 0, 1)

    #             # TODO: I'm not entirely sure whether the leaving initiation problem is fixed or not.
    #             # It probably isn't, but we can maybe hope that FA is doing a good job?
    #             valid_goal_values = goal_value_with_bonus[np.where(x_goal_init == True)]
    #             goal_target = np.max(x_goal_reward + x_goal_discount * valid_goal_values[:, np.newaxis], axis=0)
    #             x_target = np.maximum(goal_target, action_values[i])
    #             goal_targets.append(x_target)

    #             # if self.goal_termination_func(data['x'][i])[6]:
    #             #     print(f'')
    #             #     print(f'x: {data["x"][i]}')
    #             #     print(f'q: {action_values[i]} goal target: {goal_target} goal_value: {self.goal_value_learner.goal_values[6]}')

    #                 # print(f'reward {x_goal_reward} \ndiscount: {x_goal_discount} ')
    #             start_index = end_index

    #         goal_targets = np.vstack(goal_targets)
    #         data['target'] = goal_targets
    #         self.behaviour_learner.OCI_combined_update(data, polyak_stepsize=0.0005)

    def _OCI(self, sp):
        ############################ OCI
        if self.buffer.num_in_buffer >= self.min_buffer_size_before_update:
            
            # OCI. Resampling here
            OCI_batch_size = 32
            data = self.buffer.sample(OCI_batch_size)
            # Bias update towards the current state by adding it to the list of updated states.
            data['x'] = np.vstack([data['x'], sp])
            OCI_batch_size += 1
            x_goals_list = []

            goals = np.array(self.goals)
            for i in range(OCI_batch_size):
                x_goal_init = self.goal_initiation_func(data['x'][i])
                x_valid_goals = goals[np.where(x_goal_init == True)]

                tiled_x = np.tile(data['x'][i], (x_valid_goals.shape[0], 1))
                goal_x = np.append(tiled_x, x_valid_goals, axis=1)

                x_goals_list.append(goal_x)
            
            all_goal_x = np.vstack(x_goals_list)
            # _, goal_rewards, goal_discounts = self.goal_learner.get_goal_outputs(all_goal_x)

            goal_rewards = np.zeros((all_goal_x.shape[0], 5))
            goal_discounts = np.zeros((all_goal_x.shape[0], 5))
            
            goal_targets = []
            start_index = 0
            # bonus = np.where(self.tau > 12000, 1000, 0)
            bonus = 0
            # goal_value_with_bonus = np.copy(self.goal_value_learner.goal_values) + bonus
            goal_value_with_bonus = np.copy(self.goal_estimate_learner.goal_baseline)
        
            for i in range(OCI_batch_size):
                x_num_goals = x_goals_list[i].shape[0]
                end_index = start_index + x_num_goals
                x_goal_init = self.goal_initiation_func(data['x'][i])
                
                x_goal_reward = goal_rewards[start_index: end_index]
                x_goal_discount = np.clip(goal_discounts[start_index: end_index], 0, 1)

                # TODO: I'm not entirely sure whether the leaving initiation problem is fixed or not.
                # It probably isn't, but we can maybe hope that FA is doing a good job?
                valid_goal_values = goal_value_with_bonus[np.where(x_goal_init == True)]
                goal_target = np.max(x_goal_reward + x_goal_discount * valid_goal_values[:, np.newaxis], axis=0)
                goal_targets.append(goal_target)
                start_index = end_index

            OCI_data = {}
            OCI_data['x'] = data['x']
            goal_targets = np.vstack(goal_targets)
            OCI_data['target'] = goal_targets
            # print('updating OCI')
            self.behaviour_learner.OCI_update(OCI_data)

    def update(self, s: Any, a, sp: Any, r, gamma, terminal: bool = False):
        self.num_updates += 1
        self.num_steps_in_ep += 1
        if r == 10000:
            self.num_term += 1
            print(f'terminated! num_term: {self.num_term} num_steps: {self.num_steps_in_ep}')
            # print(f'goal_baseline: {self.goal_estimate_learner.goal_baseline}\ngoal_values: {self.goal_value_learner.goal_values}')
            globals.collector.collect('num_steps_in_ep', self.num_steps_in_ep)
            self.num_steps_in_ep = 0

        goal_init = self.goal_initiation_func(s)
        goal_terms = self.goal_termination_func(sp)

        # Exploration bonus:
        self.tau += 1
        self.tau[np.where(goal_terms == True)] = 0

        self._add_to_buffer(s, a, sp, r, gamma, terminal, goal_init, goal_terms)
        self._direct_rl_update()
        # self._state_to_goal_estimate_update()
        self._goal_estimate_update(s, a, sp, r, gamma, terminal, goal_init, goal_terms)
        # self._goal_value_update() # Taking out  goal value for now
        if self.OCI_update_interval > 0:
            if self.num_updates % self.OCI_update_interval == 0:
                self._OCI(sp)
            
        # Testing with combined update
        # self._OCI_combined()
  
        # # Logging
        self.cumulative_reward += r
        def log():
            globals.collector.collect('reward_rate', np.copy(self.cumulative_reward) / globals.blackboard['step_logging_interval'])
            self.cumulative_reward = 0
            # print(bonus)
        run_if_should_log(log)

        if not terminal:
            ap = self.selectAction(sp)
        else:
            ap = None

        return ap

    def agent_end(self, s, a, r, gamma):
        self.update(s, a, s, r, 0, terminal=True)

        # self.behaviour_learner.episode_end()
        # self.option_model.episode_end()

class GoalLearner_NN():
    def __init__(self, state_shape, num_actions: int, step_size: float):
        self.num_actions: int = num_actions

        # self.num_outputs: int = num_actions * 3 # adding reward and gamma output
        # self.policy_slice = slice(0, num_actions) # slice for the greedy goal reaching policy
        # self.reward_slice = slice(num_actions, num_actions * 2) # slice for the reward following greedy policy till termination
        # self.gamma_slice = slice(num_actions * 2, num_actions * 3)# slice for the discount following greedy policy till termination
        
        # Initializing jax functions
        def q_function(states):
            q_mlp = hk.Sequential([
                hk.Linear(128), jax.nn.relu,
                hk.Linear(128), jax.nn.relu,
                hk.Linear(self.num_actions),
            ])

            v_mlp = hk.Sequential([
                hk.Linear(128), jax.nn.relu,
                hk.Linear(128), jax.nn.relu,
                hk.Linear(self.num_actions),
            ])

            return q_mlp(states), v_mlp(states)

        self.f_qfunc = hk.without_apply_rng(hk.transform(q_function))

        self.f_opt = optax.adam(step_size)

        def _take_action_index(data, action):
            return jnp.take_along_axis(data, jnp.expand_dims(action, axis=1), axis=1).squeeze()
    
        def _loss(params: hk.Params, target_params: hk.Params, data):
            r = data['r']
            x = data['x']
            a = data['a']
            xp = data['xp']
            # gamma = data['gamma']
            goal_policy_cumulant = data['goal_policy_cumulant']
            goal_discount = data['goal_discount']

            policy_pred, reward_pred, gamma_pred = self.f_get_goal_output(params, x)
            # Getting values for specific actions
            policy_pred = _take_action_index(policy_pred, a)
            reward_pred = _take_action_index(reward_pred, a)
            gamma_pred = _take_action_index(gamma_pred, a)
            
            xp_policy_pred, xp_reward_pred, xp_gamma_pred = jax.lax.stop_gradient(self.f_get_goal_output(target_params, xp))
            xp_ap = jnp.argmax(xp_policy_pred, axis=1)

            policy_target = goal_policy_cumulant + goal_discount * _take_action_index(xp_policy_pred, xp_ap)
            reward_target = r + goal_discount * _take_action_index(xp_reward_pred, xp_ap)

            policy_loss = jnp.mean(jnp.square(policy_target - policy_pred))
            reward_loss = jnp.mean(jnp.square(reward_target - reward_pred))

            return policy_loss + reward_loss, (jax.lax.stop_gradient(policy_loss), jax.lax.stop_gradient(reward_loss))

        def _get_goal_output(params: hk.Params, x: Any):
            policy_output, v_output = self.f_qfunc.apply(params, x)
            # Since the problem is shortest path and we're trying to maximize the reward anyways, the signals are essentially identical.
            gamma_output = policy_output
            return policy_output, v_output, gamma_output

        def _update(params: hk.Params, target_params: hk.Params, opt_state, data):
            grads, reward_loss = jax.grad(_loss, has_aux=True)(params, target_params, data)
            updates, opt_state = self.f_opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, reward_loss
            
        self.f_get_goal_output = jax.jit(_get_goal_output)
        self.f_update = jax.jit(_update)
        # self.f_update = _update

        # Initialization
        dummy_state = jnp.zeros(state_shape)
        self.params = self.f_qfunc.init(jax.random.PRNGKey(42), dummy_state)
        self.opt_state = self.f_opt.init(self.params)

        # target params for the network
        self.target_params = copy.deepcopy(self.params)

    def get_goal_outputs(self, x: npt.ArrayLike) -> np.ndarray:
        return self.f_get_goal_output(self.params, x)

    def update(self, data, polyak_stepsize:float=0.005):
        self.params, self.opt_state, (policy_loss, reward_loss) = self.f_update(self.params, self.target_params, self.opt_state, data)
        
        def log():
            globals.collector.collect('reward_loss', reward_loss)
            globals.collector.collect('policy_loss', policy_loss)
            # print(f'{policy_loss} {reward_loss}')
        run_if_should_log(log)

        self.target_params = optax.incremental_update(self.params, self.target_params, polyak_stepsize)
        # return self.params

class QLearner_funcs():
    def __init__(self, num_actions: int, learning_rate: float, epsilon: float):
        self.num_actions = num_actions
        self.epsilon = epsilon

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
            prev_pred = jnp.take_along_axis(x_pred, jnp.expand_dims(a, axis=1), axis=1).squeeze()

            xp_pred = jax.lax.stop_gradient(jnp.max(self.f_qfunc.apply(target_params, xp), axis=1))

            # xp_values = jax.lax.stop_gradient(self.f_qfunc.apply(target_params, xp))
            # greedy_ap = np.argmax(xp_values, axis=1)   
            # policy = jax.nn.one_hot(greedy_ap, self.num_actions) * (1 - self.epsilon) + (self.epsilon / self.num_actions)
            # xp_pred = jnp.average(xp_values, axis=1, weights=policy)

            td_error = r + gamma * xp_pred - prev_pred
            return td_error

        def OCI_loss(params: hk.Params, target_params: hk.Params, data):
            target = data['target']
            x = data['x']
            x_pred = self.f_qfunc.apply(params, x)

            target = jnp.maximum(x_pred, target)
            return jnp.mean(jnp.square(target - x_pred))
            
        def loss(params: hk.Params, target_params: hk.Params, data):
            td_errors = get_td_errors(params, target_params, data)
            return  jnp.mean(jnp.square(td_errors)), td_errors

        def OCI_combined_loss(params: hk.Params, target_params: hk.Params, data):
            r = data['r']
            x = data['x']
            a = data['a']
            xp = data['xp']
            gamma = data['gamma']
            goal_target = data['target']

            x_pred = self.f_qfunc.apply(params, x)
            xp_pred = jax.lax.stop_gradient(jnp.max(self.f_qfunc.apply(target_params, xp), axis=1))
            prev_pred = jnp.take_along_axis(x_pred, jnp.expand_dims(a, axis=1), axis=1).squeeze()

            goal_target_max_action = jnp.take_along_axis(goal_target, jnp.expand_dims(a, axis=1), axis=1).squeeze()
            print(goal_target_max_action.shape)

            td_error = jnp.maximum(r + gamma * xp_pred, goal_target_max_action) - prev_pred
            return  jnp.mean(jnp.square(td_error))

        def update(params: hk.Params, target_params: hk.Params, opt_state, data):
            grads, td_errors = jax.grad(loss, has_aux=True)(params, target_params, data)
            updates, opt_state = self.f_opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, td_errors

        def OCI_update(params: hk.Params, target_params: hk.Params, opt_state, data):
            grads = jax.grad(OCI_loss)(params, target_params, data)
            updates, opt_state = self.f_opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state
        
        def OCI_combined_update(params: hk.Params, target_params: hk.Params, opt_state, data):
            grads = jax.grad(OCI_combined_loss)(params, target_params, data)
            updates, opt_state = self.f_opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state
        

        self.f_get_q_values = jax.jit(get_q_values)
        self.f_update = jax.jit(update)  
        # self.f_update = update
        self.f_get_td_errors = jax.jit(get_td_errors)  
        self.f_OCI_update = jax.jit(OCI_update)
        self.f_OCI_combined_update = jax.jit(OCI_combined_update)
        return

class QLearner_NN():
    def __init__(self, state_shape, num_actions: int, learning_rate: float, epsilon: float):
        self.num_actions: int = num_actions
        self.funcs = QLearner_funcs(num_actions, learning_rate, epsilon)

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

    def OCI_update(self, data, polyak_stepsize:float=0.005):
        self.params, self.opt_state = self.funcs.f_OCI_update(self.params, self.target_params, self.opt_state, data)
        self.target_params = optax.incremental_update(self.params, self.target_params, polyak_stepsize)
        return self.params
    
    def OCI_combined_update(self, data, polyak_stepsize:float=0.005):
        self.params, self.opt_state = self.funcs.f_OCI_combined_update(self.params, self.target_params, self.opt_state, data)
        self.target_params = optax.incremental_update(self.params, self.target_params, polyak_stepsize)
        return self.params
    
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
                # # if g == 12:
                # #     print(goal_init)
                self.r[g] += alpha * (np.sum(option_pi_x * r_s, axis=1) - self.r[g])
                self.gamma[g] += alpha * (np.sum(option_pi_x * gamma_s, axis=1)- self.gamma[g])
                self.goal_baseline[g] += alpha * (np.max(x_action_values) - self.goal_baseline[g])
                self.goal_init[g] += alpha * (goal_init - self.goal_init[g])

        def log():
            globals.collector.collect('goal_r', np.copy(self.r))
            globals.collector.collect('goal_gamma', np.copy(self.gamma))
            globals.collector.collect('goal_baseline', np.copy(self.goal_baseline))
            # print(self.goal_baseline)
            globals.collector.collect('goal_init', np.copy(self.goal_init))
        run_if_should_log(log)

class GoalValueLearner:
    def __init__(self, num_goals):
        self.num_goals = num_goals
        
        # Initializing goal values
        self.goal_values = np.zeros(self.num_goals)
    
    def update(self, goal_gamma, reward_goals, goal_bonus, goal_init, goal_baseline):
        num_planning_steps = 1
        for _ in range(num_planning_steps):
            # Just doing value iteration for now 

            # Can we possibly vectorize this?
            for g in range(self.num_goals):
                # print(goal_gamma[g])
                returns = reward_goals[g] + goal_gamma[g] * (self.goal_values + goal_bonus)
                valid_goals = np.nonzero(goal_init[g])

                if len(valid_goals[0]) > 0:
                    self.goal_values[g] = max(goal_baseline[g], np.max(returns[valid_goals[0]]))
                    # print(returns[valid_goals[0]], goal_baseline[g])
                else:
                    self.goal_values[g] = goal_baseline[g]
        
        if globals.blackboard['num_steps_passed'] % globals.blackboard['step_logging_interval'] == 0:
            globals.collector.collect('goal_values', np.copy(self.goal_values)) 