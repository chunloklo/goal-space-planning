
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
from typing import Dict, Optional, Union, Tuple, Any, TYPE_CHECKING
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

if TYPE_CHECKING:
    # Important for forward reference
    from src.problems.BaseProblem import BaseProblem

# [2022-03-09 chunlok] Copied over from DynaOptions_NN. For now, getting this to work with Pinball env.
class Dyna_NN:
    def __init__(self, problem: 'BaseProblem'):
        self.wrapper_class = rlglue.OneStepWrapper
        self.env = problem.getEnvironment()
        self.num_actions = problem.actions
        self.params = problem.params
        self.random = np.random.RandomState(problem.seed)
        
        self.step_size = 1e-3
        self.batch_size = 16
        self.epsilon = self.params['epsilon']

        self.behaviour_learner = QLearner_NN((4,), 5, self.step_size)
        self.goal_policy_learners = []

        self.buffer_size = 1000000
        self.min_buffer_size_before_update = 1000
        self.buffer = Buffer(self.buffer_size, {'x': (4,), 'a': (), 'xp': (4,), 'r': (), 'gamma': ()}, self.random.randint(0,2**31))

        self.num_steps_in_ep = 0
        
        params = self.params
        self.use_pretrained_behavior = param_utils.parse_param(params, 'use_pretrained_behavior', lambda p : isinstance(p, bool), optional=True, default=False)
        self.polyak_stepsize = param_utils.parse_param(params, 'polyak_step_size', lambda p : isinstance(p, float) and p >= 0)
        print(self.polyak_stepsize)

        if self.use_pretrained_behavior:
            self.behaviour_learner = pickle.load(open('src/environments/data/pinball/behavior_learner.pkl', 'rb'))
            print('using pretrained behavior')

        self.cumulative_reward = 0

        self.num_term = 0

    def FA(self):
        return "Neural Network"

    def __str__(self):
        return "Dyna_NN"

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
        self.behaviour_learner.update(data, polyak_stepsize=self.polyak_stepsize)

    def update(self, s: Any, a, sp: Any, r, gamma, terminal: bool = False):
        self.buffer.update({'x': s, 'a': a, 'xp': sp, 'r': r, 'gamma': gamma})
        self.num_steps_in_ep += 1

        if r == 10000:
            self.num_term += 1
            # print(f'terminated! term_number: {self.num_term}')
            globals.collector.collect('num_steps_in_ep', self.num_steps_in_ep)
            self.num_steps_in_ep = 0
        # if terminal:
            # print('terminal')

        # if s == [0.2, 0.9, 0.0, 0.0]:
        #     print(f'{self.behaviour_learner.get_action_values(np.array(s))}')


        self.update_behavior()
                # print(s)
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
        # self.behaviour_learner.episode_end()
        # self.option_model.episode_end()

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
            
        def loss(params: hk.Params, target_params: hk.Params, data):
            td_errors = get_td_errors(params, target_params, data)
            return  jnp.mean(jnp.square(td_errors)), td_errors

        def update(params: hk.Params, target_params: hk.Params, opt_state, data):
            grads, td_errors = jax.grad(loss, has_aux=True)(params, target_params, data)
            updates, opt_state = self.f_opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, td_errors

        self.f_get_q_values = jax.jit(get_q_values)
        self.f_update = jax.jit(update)  
        self.f_get_td_errors = jax.jit(get_td_errors)      
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