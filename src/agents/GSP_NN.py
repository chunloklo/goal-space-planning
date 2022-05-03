
from functools import partial
import numpy as np
from tqdm import tqdm
from src.utils import rlglue
from src.utils import globals
from src.utils import options, param_utils
from src.utils.param_utils import parse_param
import numpy.typing as npt
from PyFixedReps.Tabular import Tabular
from typing import Dict, List, Optional, Union, Tuple, Any, TYPE_CHECKING
import jax.numpy as jnp
from src.agents.components.buffers import Buffer, DictBuffer
import jax
import haiku as hk
import optax
import copy
import pickle
from .components.QLearner_NN import QLearner_NN
from .components.EQRC_model import GoalLearner_EQRC_NN, GoalLearner_QRC_NN
from .components.DQN_model import GoalLearner_DQN_NN
from .components.EQRC_NN import EQRC_NN
import cloudpickle

from src.utils.log_utils import get_last_pinball_action_value_map, run_if_should_log
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
        self.problem = problem

        # Initializing goal information
        self.goals = problem.goals
        self.num_goals = self.goals.num_goals
        self.goal_termination_func = self.goals.goal_termination
        self.goal_initiation_func = self.goals.goal_initiation

        def modified_goal_init(xs):  
            init =  self.goals.goal_initiation(xs)
            return init
        
        self.goal_initiation_func = modified_goal_init

        params = self.params
        
        # BEHAVIOUR PARAMS
        # Controls the number of samples to sample from the buffer when performing an update 
        self.batch_size = parse_param(params, 'batch_size', lambda p : isinstance(p, int) and p > 0)
        self.batch_num = parse_param(params, 'batch_num', lambda p : isinstance(p, int) and p > 0)
        self.behaviour_alg = parse_param(params, 'behaviour_alg', lambda p : p in ['DQN', 'QRC'])
        # Step size for both the behaviour and model learner. These are combined for now
        self.step_size = parse_param(params, 'step_size', lambda p : isinstance(p, float) and p >= 0.0) 
        # Epsilon for epsilon-greedy policy
        self.epsilon = parse_param(params, 'epsilon', lambda p : isinstance(p, float) and p >= 0.0)

        # Goal learner params
        self.goal_learner_step_size = parse_param(params, 'goal_learner_step_size', lambda p : isinstance(p, float) and p >= 0.0) 

        # GOAL ESTIMATE PARAMS
        self.goal_estimate_batch_size = parse_param(params, 'goal_estimate_batch_size', lambda p : isinstance(p, int) and p > 0)
        self.goal_estimate_update_interval = parse_param(params, 'goal_estimate_update_interval', lambda p : isinstance(p, int) and p > 0)
        self.goal_estimate_step_size = parse_param(params, 'goal_estimate_step_size', lambda p : isinstance(p, float) and p >= 0.0) 

        self.use_oci_target_update = parse_param(params, 'use_oci_target_update', lambda p : isinstance(p, bool), optional=True, default=False)
        self.oci_beta = parse_param(params, 'oci_beta', lambda p : isinstance(p, float) and p >= 0.0 and p <= 1.0, optional=True, default=0.5)

        if not self.use_oci_target_update:
            self.oci_update_interval = parse_param(params, 'oci_update_interval', lambda p : isinstance(p, int) and p >= 0) # Number of update steps between each OCI update
            self.oci_batch_size = parse_param(params, 'oci_batch_size', lambda p : isinstance(p, int) and p > 0)
            self.oci_batch_num = parse_param(params, 'oci_batch_num', lambda p : isinstance(p, int) and p > 0)
        
        # Optional parameters
        # Whether to use the baseline or goal values for OCI
        self.use_goal_values = parse_param(params, 'use_goal_values', lambda p : isinstance(p, bool)) 
        # Whether the agent should only learn the model, and not learn the behaviour
        self.learn_model_mode = parse_param(params, 'learn_model_mode', lambda p : p in ['only', 'online', 'fixed'])
        # Additional parameter for controlling adam's epsilon parameter for model learning
        self.adam_eps = parse_param(params, 'adam_eps', lambda p : isinstance(p, float) and p >= 0, optional=True, default=1e-8)
        # Whether to use goal-based exploration bonus or not
        self.use_exploration_bonus = parse_param(params, 'use_exploration_bonus', lambda p : isinstance(p, bool), optional=True, default=False)

        # Pretrain goal values
        self.save_pretrain_goal_values = parse_param(params, 'save_pretrain_goal_values', lambda p : isinstance(p, str) or p is None, optional=True, default=None) 
        self.load_pretrain_goal_values = parse_param(params, 'load_pretrain_goal_values', lambda p : isinstance(p, str) or p is None, optional=True, default=None) 
        self.use_pretrained_goal_values_optimization = parse_param(params, 'use_pretrained_goal_values_optimization', lambda p : isinstance(p, bool), optional=True, default=False) 
        self.load_behaviour_as_goal_values = parse_param(params, 'load_behaviour_as_goal_values', lambda p : isinstance(p, str) or p is None, optional=True, default=None) 
        self.behaviour_goal_value_mode = parse_param(params, 'behaviour_goal_value_mode', lambda p : p in ['direct', 'only_values'] or p is None, optional=True, default=None) 

        if self.load_behaviour_as_goal_values:
            assert self.behaviour_goal_value_mode is not None
        if self.behaviour_goal_value_mode is not None:
            assert self.load_behaviour_as_goal_values is not None

        if self.use_exploration_bonus:
            self.exploration_bonus_amount = parse_param(params, 'exploration_bonus_amount', lambda p : isinstance(p, float) and p >= 0)

        # The amount of time to simply add to the buffer and not learn anything 
        self.prefill_buffer_time = parse_param(params, 'prefill_buffer_time', lambda p : isinstance(p, int) and p >= 0, optional=True, default=0)
        # Whether to only train the state to goal models on specific goals.
        # For now, the list is a tuple since lists aren't hashable.
        self.learn_select_goal_models = parse_param(params, 'learn_select_goal_models', lambda p : isinstance(p, tuple) or p is None, optional=True, default=None)

        # Name of the file that contains the pretrained behavior that the agent should load. If None, its starts from scratch
        self.load_behaviour_name = parse_param(params, 'load_behaviour_name', lambda p : isinstance(p, str) or p is None, optional=True, default=None)
        # Name for the pre-trained model
        self.pretrained_model_name = parse_param(params, 'pretrained_model_name', lambda p : isinstance(p, str) or p is None, optional=True, default=None)
        # Name for the pre-filled buffer
        self.prefill_goal_buffer = parse_param(params, 'prefill_buffer_name', lambda p : isinstance(p, str) or p is None, optional=True, default=None)
        # Whether to only use experimence within goal boundaries for model learning
        self.use_goal_boundaries_for_model_learning = parse_param(params, 'use_goal_boundaries_for_model_learning', lambda p : isinstance(p, bool), optional=True, default=True)
        
        # Some fixed parameters that might want to get parameterized later
        self.buffer_size = 1000000
        self.min_buffer_size_before_update = 10000
        self.goal_min_buffer_size_before_update = 1000
        # self.min_buffer_size_before_update = 1000

        # Hard coding this for the environment for now
        self.obs_shape = (4, )
        
        self.goal_value_learner = GoalValueLearner(self.num_goals, self.problem.terminal_goal_index)
        if self.behaviour_alg == 'QRC':
            self.beta = parse_param(params, 'beta', lambda p : isinstance(p, float), optional=True, default=1.0)
            self.behaviour_learner = EQRC_NN((4,), self.num_actions, self.step_size, self.epsilon, beta=self.beta)
        elif self.behaviour_alg == 'DQN':
            self.polyak_stepsize = parse_param(params, 'polyak_stepsize', lambda p : isinstance(p, float) and p >= 0)
            self.behaviour_learner = QLearner_NN((4,), 5, self.step_size, self.polyak_stepsize, self.oci_beta)
        
        if self.use_pretrained_goal_values_optimization:
            self.buffer = Buffer(self.buffer_size, {
                'x': self.obs_shape, 
                'a': (), 
                'xp': self.obs_shape, 
                'r': (), 
                'gamma': (), 
                'goal_inits': (self.num_goals, ), 
                'goal_terms': (self.num_goals, ), 
                'target': ()},
            self.random.randint(0,2**31), 
            type_map={'a': np.int64})

            # batching buffer add such that we can batch together goal model calls
            self.batch_buffer_add_size = parse_param(params, 'batch_buffer_add_size', lambda p : isinstance(p, int) and p > 0)
            self.intermediate_buffer = Buffer(self.batch_buffer_add_size, {
                'x': self.obs_shape, 
                'a': (), 
                'xp': self.obs_shape, 
                'r': (), 
                'gamma': (), 
                'goal_inits': (self.num_goals, ), 
                'goal_terms': (self.num_goals, )},
            self.random.randint(0,2**31), 
            type_map={'a': np.int64})

        else:
            self.buffer = Buffer(self.buffer_size, {
                'x': self.obs_shape, 'a': (), 'xp': self.obs_shape, 'r': (), 'gamma': (), 'goal_inits': (self.num_goals, ), 'goal_terms': (self.num_goals, )}, 
                self.random.randint(0,2**31), 
                type_map={'a': np.int64})

        self.goal_estimate_buffer = Buffer(self.buffer_size, 
            {'xp': self.obs_shape, 'goal_inits': (self.num_goals, ), 'goal_terms': (self.num_goals, )},
            self.random.randint(0,2**31))

        self.goal_buffers = []
        for _ in range(self.num_goals):
            self.goal_buffers.append(Buffer(self.buffer_size, 
                {'x': self.obs_shape, 'a': (), 'xp': self.obs_shape, 'r': (), 'gamma': (), 'goal_policy_cumulant': (), 'goal_discount': ()}, 
                self.random.randint(0,2**31), 
                type_map={'a': np.int64}))

        self.save_buffer_name = parse_param(params, 'save_buffer_name', lambda p : isinstance(p, str) or p is None, optional=True, default=None)
        self.load_buffer_name = parse_param(params, 'load_buffer_name', lambda p : isinstance(p, str) or p is None, optional=True, default=None)
        
        self.goal_estimate_learner = GoalEstimates(self.num_goals)
        # self.goal_learners = [GoalLearner_EQRC_NN(self.obs_shape, self.num_actions, self.goal_learner_step_size, 0.1, beta=1.0) for _ in range(self.num_goals)]
        # self.goal_learners = [GoalLearner_QRC_NN(self.obs_shape, self.num_actions, self.goal_learner_step_size, beta=1.0) for _ in range(self.num_goals)]
        self.goal_learners = [GoalLearner_DQN_NN(self.obs_shape, self.num_actions, self.goal_learner_step_size, 0.1, self.polyak_stepsize, self.adam_eps) for _ in range(self.num_goals)]

        if self.load_behaviour_name is not None:
            agent = pickle.load(open(f'src/environments/data/pinball/{self.load_behaviour_name}_agent.pkl', 'rb'))
            self.behaviour_learner = agent.behaviour_learner
            # self.buffer = agent.buffer
            # self.goal_learners = agent.goal_learners
            # self.goal_estimate_learner = agent.goal_estimate_learner
            # self.goal_buffers = agent.goal_buffers
            # self.goal_value_learner = agent.goal_value_learner

        # [chunlok 20202-04-15] TODO The specific path of these saved models might need to be changed later to be more general
        if self.pretrained_model_name is not None:
            self.goal_learners = pickle.load(open(f'./src/environments/data/pinball/{self.pretrained_model_name}_goal_learner.pkl', 'rb'))
            self.goal_buffers = pickle.load(open(f'./src/environments/data/pinball/{self.pretrained_model_name}_goal_buffer.pkl', 'rb'))
            self.goal_estimate_buffer = pickle.load(open(f'./src/environments/data/pinball/{self.pretrained_model_name}_goal_estimate_buffer.pkl', 'rb'))
            print_str = 'Num in goal buffers: '
            for g in range(self.num_goals):
                print_str += f'{self.goal_buffers[g].num_in_buffer}, '
            print(print_str)
            print(f'Num in goal estimate buffer: {self.goal_estimate_buffer.num_in_buffer}')

        self.prefill_goal_buffer = parse_param(params, 'use_prefill_goal_buffer', lambda p : isinstance(p, str) or p is None, optional=True, default=None)
        if self.prefill_goal_buffer is not None:
            self.goal_buffers = pickle.load(open(f'./src/environments/data/pinball/{self.prefill_goal_buffer}_goal_buffer.pkl', 'rb'))

            possible = []
            for g in range(self.num_goals):
                possible.append(self.goal_buffers[g].num_in_buffer)
                #     possible.append(True)
                # else:
                #     possible.append(False)
            print(f'Size of each goal buffer: {possible}')
        
        if self.load_pretrain_goal_values is not None:
            self.goal_estimate_learner = pickle.load(open(f'./src/environments/data/pinball/{self.load_pretrain_goal_values}_pretrain_goal_estimate_learner.pkl', 'rb'))
            self.goal_value_learner = pickle.load(open(f'./src/environments/data/pinball/{self.load_pretrain_goal_values}_pretrain_goal_value_learner.pkl', 'rb'))
            print(f'pretrained goal values: {self.goal_value_learner.goal_values}')

        if self.load_behaviour_as_goal_values is not None:
            agent = pickle.load(open(f'src/environments/data/pinball/{self.load_behaviour_as_goal_values}_agent.pkl', 'rb'))
            self.behaviour_goal_value = agent.behaviour_learner

        if self.load_buffer_name is not None:
            self.buffer = pickle.load(open(f'./src/environments/data/pinball/{self.load_buffer_name}_buffer.pkl', 'rb'))
            print(f'loaded buffer with size: {self.buffer.num_in_buffer}')

        self.cumulative_reward = 0
        self.num_term = 0
        self.num_updates = 0
        self.num_steps_in_ep = 0

        self.tau = np.full(self.num_goals, 1)

        if self.save_pretrain_goal_values is not None:  
            self._pretrain_goal_values()

    def FA(self):
        return "Neural Network"

    def __str__(self):
        return "GSP_NN"

    def get_policy(self, s) -> npt.ArrayLike:

        if self.epsilon == 1.0:
            return np.full(self.num_actions, 1.0 / self.num_actions)
        
        action_values = self.behaviour_learner.get_action_values(s)
        # print(action_values)
        # epsilon greedy
        a = np.argmax(action_values)
            
        policy = np.zeros(self.num_actions)
        policy += self.epsilon / (self.num_actions)
        policy[a] += 1 - self.epsilon
        return policy

    # public method for rlglue
    def selectAction(self, s: Any) -> int:
        s = np.array(s)
        a = self.random.choice(self.num_actions, p = self.get_policy(s))
        return a

    def _get_behaviour_goal_values(self, xs):
        batch_size = xs.shape[0]
        targets = np.array(self.behaviour_goal_value.get_action_values(xs))

        # Masking out invalid goals based on the initiation func
        for i in range(batch_size):
            x_goal_init = self.goal_initiation_func(xs[i])
            if np.all(~x_goal_init):
                targets[i] = np.nan

        return np.max(targets, axis=1)

    def _log_model_error_heatmap(self):
        if not globals.collector.has_key('model_error_heatmap'):
            return

        def get_error(s):
            batch_size = 1
            xs = np.array([s])
            goal_states = np.hstack((self.goals.goals, self.goals.goal_speeds))
            goal_dest_values = np.array(self.behaviour_goal_value.get_action_values(goal_states))
            goal_dest_values = np.max(goal_dest_values, axis=1)

            goal_r = np.empty((batch_size, self.num_goals, self.num_actions))
            goal_gammas = np.empty((batch_size, self.num_goals, self.num_actions))
            # The goal policy is not used right now
            goal_policy_q = np.empty((batch_size, self.num_goals, self.num_actions)) 

            for g in range(self.num_goals):
                goal_policy_q[:, g, :], goal_r[:, g, :], goal_gammas[:, g, :] = self.goal_learners[g].get_goal_outputs(xs)
            pass

            # Getting one-hot policies for the goal policies
            goal_policies = np.zeros((batch_size, self.num_goals, self.num_actions))
            np.put_along_axis(goal_policies, np.expand_dims(np.argmax(goal_policy_q, axis=2), -1), 1, axis=2)

            goal_r = np.sum(goal_r * goal_policies, axis=2)
            goal_gammas = np.sum(goal_gammas * goal_policies, axis=2)

            goal_gammas = np.clip(goal_gammas, 0, 1)
            
            goal_values = goal_r + goal_gammas * goal_dest_values

            # Masking out invalid goals based on the initiation func
            for i in range(batch_size):
                x_goal_init = self.goal_initiation_func(xs[i])
                invalid_goals = np.where(x_goal_init == False)[0]
                goal_values[i, invalid_goals] = np.nan
            targets = np.nanmax(goal_values, axis=1)

            ##### Mainly checking for errors
            oracle_goal_values = self._get_behaviour_goal_values(xs)

            return np.nanmean(np.square(oracle_goal_values - targets))

        RESOLUTION = 40
        last_goal_q_map = np.zeros((RESOLUTION, RESOLUTION, self.num_actions))
        goal_action_value = get_last_pinball_action_value_map(1, get_error, resolution=RESOLUTION)
        last_goal_q_map = goal_action_value[0]

        globals.collector.collect('model_error_heatmap', last_goal_q_map)
        pass

    def _log_model_error(self):
        if not globals.collector.has_key('model_error'):
            return
    
        def log():
            batch_size = 1024
            if batch_size > self.buffer.num_in_buffer:
                globals.collector.collect('model_error', 0)
                return

            data = self.buffer.sample(batch_size)
            xs = data['xp']

            batch_size = xs.shape[0]
            goal_states = np.hstack((self.goals.goals, self.goals.goal_speeds))
            goal_dest_values = np.array(self.behaviour_goal_value.get_action_values(goal_states))
            goal_dest_values = np.max(goal_dest_values, axis=1)

            goal_r = np.empty((batch_size, self.num_goals, self.num_actions))
            goal_gammas = np.empty((batch_size, self.num_goals, self.num_actions))
            # The goal policy is not used right now
            goal_policy_q = np.empty((batch_size, self.num_goals, self.num_actions)) 

            for g in range(self.num_goals):
                goal_policy_q[:, g, :], goal_r[:, g, :], goal_gammas[:, g, :] = self.goal_learners[g].get_goal_outputs(xs)
            pass

            # Getting one-hot policies for the goal policies
            goal_policies = np.zeros((batch_size, self.num_goals, self.num_actions))
            np.put_along_axis(goal_policies, np.expand_dims(np.argmax(goal_policy_q, axis=2), -1), 1, axis=2)

            goal_r = np.sum(goal_r * goal_policies, axis=2)
            goal_gammas = np.sum(goal_gammas * goal_policies, axis=2)

            goal_gammas = np.clip(goal_gammas, 0, 1)
            
            goal_values = goal_r + goal_gammas * goal_dest_values

            # Masking out invalid goals based on the initiation func
            for i in range(batch_size):
                x_goal_init = self.goal_initiation_func(xs[i])
                invalid_goals = np.where(x_goal_init == False)[0]
                goal_values[i, invalid_goals] = np.nan
            targets = np.nanmax(goal_values, axis=1)

            ##### Mainly checking for errors
            oracle_goal_values = self._get_behaviour_goal_values(xs)

            mse = np.nanmean(np.square(oracle_goal_values - targets))

            globals.collector.collect('model_error', mse)

        # def log():
        #     batch_size = 1024
        #     if batch_size > self.buffer.num_in_buffer:
        #         globals.collector.collect('model_error', 0)
        #         return

        #     data = self.buffer.sample(batch_size)
        #     xs = data['xp']

        #     batch_size = xs.shape[0]
        #     goal_states = np.hstack((self.goals.goals, self.goals.goal_speeds))
        #     goal_dest_values = np.array(self.behaviour_goal_value.get_action_values(goal_states))
        #     goal_dest_values = np.max(goal_dest_values, axis=1)

        #     goal_r = np.empty((batch_size, self.num_goals, self.num_actions))
        #     goal_gammas = np.empty((batch_size, self.num_goals, self.num_actions))
        #     # The goal policy is not used right now
        #     goal_policy_q = np.empty((batch_size, self.num_goals, self.num_actions)) 

        #     for g in range(self.num_goals):
        #         goal_policy_q[:, g, :], goal_r[:, g, :], goal_gammas[:, g, :] = self.goal_learners[g].get_goal_outputs(xs)
        #     pass
        #     goal_gammas = np.clip(goal_gammas, 0, 1)
        #     goal_values = goal_r + goal_gammas * goal_dest_values[None, :, None]

        #     # Masking out invalid goals based on the initiation func
        #     for i in range(batch_size):
        #         x_goal_init = self.goal_initiation_func(xs[i])
        #         invalid_goals = np.where(x_goal_init == False)[0]
        #         goal_values[i, invalid_goals] = np.nan

        #     print(np.nanmax(goal_values, axis=2).shape)
        #     best_goals = np.nanargmax(np.nanmax(goal_values, axis=2), axis=1)
        #     targets = goal_values[:, best_goals]


        #     print(targets.shape)

        #     ##### Mainly checking for errors
        #     oracle_goal_values = np.array(self.behaviour_goal_value.get_action_values(xs))

        #     # Masking out invalid goals based on the initiation func
        #     for i in range(batch_size):
        #         x_goal_init = self.goal_initiation_func(xs[i])
        #         if np.all(~x_goal_init):
        #             oracle_goal_values[i] = np.nan

        #     print(oracle_goal_values.shape)
        #     mse = np.nanmean(np.square(oracle_goal_values - targets))

        #     globals.collector.collect('model_error', mse)

        run_if_should_log(log)

    def _get_best_goal_values(self, xs, action_values: bool = False):
        batch_size = xs.shape[0]

        if self.behaviour_goal_value_mode == 'direct':
            assert not action_values
            return self._get_behaviour_goal_values(xs)
        
        if self.behaviour_goal_value_mode == 'only_values':
            # self.goals.goal
            assert not action_values
            goal_states = np.hstack((self.goals.goals, self.goals.goal_speeds))
            goal_dest_values = np.array(self.behaviour_goal_value.get_action_values(goal_states))
            goal_dest_values = np.max(goal_dest_values, axis=1)

            goal_r = np.empty((batch_size, self.num_goals, self.num_actions))
            goal_gammas = np.empty((batch_size, self.num_goals, self.num_actions))
            # The goal policy is not used right now
            goal_policy_q = np.empty((batch_size, self.num_goals, self.num_actions)) 

            for g in range(self.num_goals):
                goal_policy_q[:, g, :], goal_r[:, g, :], goal_gammas[:, g, :] = self.goal_learners[g].get_goal_outputs(xs)
            pass

            # Getting one-hot policies for the goal policies
            goal_policies = np.zeros((batch_size, self.num_goals, self.num_actions))
            np.put_along_axis(goal_policies, np.expand_dims(np.argmax(goal_policy_q, axis=2), -1), 1, axis=2)

            goal_r = np.sum(goal_r * goal_policies, axis=2)
            goal_gammas = np.sum(goal_gammas * goal_policies, axis=2)

            goal_gammas = np.clip(goal_gammas, 0, 1)

            # print(goal_r.shape)
            # print(goal_gammas.shape)
            # print(goal_dest_values.shape)

            # print(f'goal_r {goal_r[0]}')
            # print(f'goal_gammas {goal_gammas[0]}')
            # print(f'dest_values {goal_dest_values}')
            
            goal_values = goal_r + goal_gammas * goal_dest_values

            # Masking out invalid goals based on the initiation func
            for i in range(batch_size):
                x_goal_init = self.goal_initiation_func(xs[i])
                invalid_goals = np.where(x_goal_init == False)[0]
                goal_values[i, invalid_goals] = np.nan

                # Making it so it doesn't bootstrap to anything other than the final state if at terminal goal
                x_goal_term = self.goal_termination_func(None, None, xs[i])
                if x_goal_term[0]:
                    goal_values[i, :] = np.nan

            targets = np.nanmax(goal_values, axis=1)

            print(goal_values[0])
            print(targets[0])
            return targets


            

        bonus = self._get_exploration_bonus()
        if self.use_goal_values:
            goal_value_with_bonus = np.copy(self.goal_value_learner.goal_values) + bonus
        else:
            goal_value_with_bonus = np.copy(self.goal_estimate_learner.goal_baseline) + bonus

        goal_r = np.empty((batch_size, self.num_goals, self.num_actions))
        goal_gammas = np.empty((batch_size, self.num_goals, self.num_actions))
        # The goal policy is not used right now
        goal_policy_q = np.empty((batch_size, self.num_goals, self.num_actions)) 

        for g in range(self.num_goals):
            goal_policy_q[:, g, :], goal_r[:, g, :], goal_gammas[:, g, :] = self.goal_learners[g].get_goal_outputs(xs)

        goal_gammas = np.clip(goal_gammas, 0, 1)

        if not action_values:
            # Getting one-hot policies for the goal policies
            goal_policies = np.zeros((batch_size, self.num_goals, self.num_actions))
            np.put_along_axis(goal_policies, np.expand_dims(np.argmax(goal_policy_q, axis=2), -1), 1, axis=2)

            goal_r = np.sum(goal_r * goal_policies, axis=2)
            goal_gammas = np.sum(goal_gammas * goal_policies, axis=2)

            targets = goal_r + goal_gammas * goal_value_with_bonus[None, :]
        else:
            targets = goal_r + goal_gammas * goal_value_with_bonus[None, :, None]

        # Masking out invalid goals based on the initiation func
        for i in range(batch_size):
            x_goal_init = self.goal_initiation_func(xs[i])
            invalid_goals = np.where(x_goal_init == False)[0]
            targets[i, invalid_goals] = np.nan

        max_goal = np.nanmax(targets, axis=1)
        return max_goal

    def _add_to_buffer(self, s, a, sp, r, gamma, terminal, goal_inits, goal_terms):

        if self.learn_model_mode != 'only':
            if self.use_pretrained_goal_values_optimization:
                self.intermediate_buffer.update({'x': s, 'a': a, 'xp': sp, 'r': r, 'gamma': gamma, 'goal_inits': goal_inits, 'goal_terms': goal_terms})

                if self.num_updates % self.batch_buffer_add_size == self.batch_buffer_add_size - 1:
                    _x = self.intermediate_buffer.buffer['x']
                    _a = self.intermediate_buffer.buffer['a']
                    _xp = self.intermediate_buffer.buffer['xp']
                    _r = self.intermediate_buffer.buffer['r']
                    _gamma = self.intermediate_buffer.buffer['gamma']
                    _goal_inits = self.intermediate_buffer.buffer['goal_inits']
                    _goal_terms = self.intermediate_buffer.buffer['goal_terms']
                    _targets = self._get_best_goal_values(_xp)

                    for i in range(self.batch_buffer_add_size):
                        self.buffer.update({'x': _x[i], 'a': _a[i], 'xp': _xp[i], 'r': _r[i], 'gamma': _gamma[i], 'goal_inits': _goal_inits[i], 'goal_terms': _goal_terms[i], 'target': _targets[i]})
            else:
                self.buffer.update({'x': s, 'a': a, 'xp': sp, 'r': r, 'gamma': gamma, 'goal_inits': goal_inits, 'goal_terms': goal_terms})

        if np.any(goal_terms):
            self.goal_estimate_buffer.update({'xp': sp, 'goal_inits': goal_inits, 'goal_terms': goal_terms})
        
        sp_goal_init = self.goal_initiation_func(sp)

        goal_discount = np.empty(goal_terms.shape)
        goal_discount[goal_terms == True] = 0
        goal_discount[goal_terms == False] = gamma

        goal_policy_cumulant = np.empty(goal_terms.shape)
        goal_policy_cumulant[goal_terms == True] = gamma
        goal_policy_cumulant[goal_terms == False] = 0

        zeros = np.zeros(goal_terms.shape)

        for g in range(self.num_goals):
            if self.use_goal_boundaries_for_model_learning:
                # If the experience takes you outside the initiation zone, then the policy definitely doesn't want to go that way.
                if goal_inits[g]:
                    if goal_terms[g] or sp_goal_init[g] != False:
                        self.goal_buffers[g].update({'x': s, 'a': a, 'xp': sp, 'r': r, 'gamma': gamma, 'goal_policy_cumulant': goal_policy_cumulant[g], 'goal_discount': goal_discount[g]})
                    else:
                        self.goal_buffers[g].update({'x': s, 'a': a, 'xp': sp, 'r': 0, 'gamma': 0, 'goal_policy_cumulant': zeros[g], 'goal_discount': zeros[g]})

    def _behaviour_update(self):
        if self.buffer.num_in_buffer < self.min_buffer_size_before_update:
            # Not enough samples in buffer to update yet
            return 

        # Updating behavior
        for _ in range(self.batch_num):
            data = self.buffer.sample(self.batch_size)
            self.behaviour_learner.update(data)

    def _pretrain_goal_values(self):
        # Updating goal estimates
        
        iter = range(5000)
        if globals.aux_config.get('show_progress'):
            iter = tqdm(iter)
        for _ in iter:
            # There is logic inside _goal_estimate_update that could prevent updating.
            # However, since self.num_steps should be 0 when initializing, the update should run given there's 
            # a sufficient number of samples inside.
            self._goal_estimate_update()
        print(f'gamma: {self.goal_estimate_learner.gamma}')
        print(f'r: {self.goal_estimate_learner.r}')
        
        self.goal_estimate_learner.goal_baseline[self.problem.terminal_goal_index] = 20000

        for _ in range(1000):
            self._goal_value_update()
        print(f'pretrained_goal_values: {self.goal_value_learner.goal_values}')
 
        cloudpickle.dump(self.goal_estimate_learner, open(f'./src/environments/data/pinball/{self.save_pretrain_goal_values}_pretrain_goal_estimate_learner.pkl', 'wb'))
        cloudpickle.dump(self.goal_value_learner, open(f'./src/environments/data/pinball/{self.save_pretrain_goal_values}_pretrain_goal_value_learner.pkl', 'wb'))

    def _oci_target_update(self):
        if self.buffer.num_in_buffer < self.min_buffer_size_before_update:
            # Not enough samples in buffer to update yet
            return 

        # Updating behavior
        for _ in range(self.batch_num):
            data = self.buffer.sample(self.batch_size)

            if self.use_pretrained_goal_values_optimization:
                self.behaviour_learner.oci_target_update(data)
                # Just do this and ignore the rest
                continue

            data['target'] = self._get_best_goal_values(data['xp'])
            self.behaviour_learner.oci_target_update(data)

    def _state_to_goal_estimate_update(self):
        # for g in range(self.num_goals):
        iter = self.learn_select_goal_models if self.learn_select_goal_models is not None else range(self.num_goals)
        for g in iter:
            if self.goal_buffers[g].num_in_buffer >= self.goal_min_buffer_size_before_update:
                batch_num = 1
                for _ in range(batch_num):
                    data = self.goal_buffers[g].sample(self.batch_size, copy=False)
                    self.goal_learners[g].update(data) 

    def _goal_estimate_update(self):
        if self.num_updates % self.goal_estimate_update_interval != 0:
            return
            
        batch_size = self.goal_estimate_batch_size
        if self.goal_estimate_buffer.num_in_buffer < batch_size:
            return 
            
        data = self.goal_estimate_buffer.sample(batch_size)
        
        sps = data['xp']

        goal_r = np.empty((batch_size, self.num_goals, self.num_actions))
        goal_gammas = np.empty((batch_size, self.num_goals, self.num_actions))
        goal_policy_q = np.empty((batch_size, self.num_goals, self.num_actions))

        for g in range(self.num_goals):
            goal_policy_q[:, g, :], goal_r[:, g, :], goal_gammas[:, g, :] = self.goal_learners[g].get_goal_outputs(sps)

        # Getting one-hot policies for the goal policies
        goal_policies = np.zeros((batch_size, self.num_goals, self.num_actions))
        np.put_along_axis(goal_policies, np.expand_dims(np.argmax(goal_policy_q, axis=2), -1), 1, axis=2)

        # Should we be using the target action values here or the actual action values?
        # action_values = self.behaviour_learner.get_target_action_values(sps)
        action_values = self.behaviour_learner.get_action_values(sps)

        data['goal_rs'] = goal_r
        data['goal_gammas'] = np.clip(goal_gammas, 0, 1) # Clipping between 0 and 1 here
        data['option_pi_x'] = goal_policies
        data['action_values'] = action_values

        self.goal_estimate_learner.batch_update(data, self.goal_estimate_step_size)

    def _get_exploration_bonus(self):
        if self.use_exploration_bonus:
            return np.where(self.tau >= 1, self.exploration_bonus_amount, 0)
        else:
            return np.zeros(self.num_goals)

    def _goal_value_update(self):
        self.goal_value_learner.update(
            self.goal_estimate_learner.gamma, 
            self.goal_estimate_learner.r, 
            self._get_exploration_bonus(), 
            self.goal_estimate_learner.goal_init, 
            self.goal_estimate_learner.goal_baseline)

    def _oci(self, sp):
        if self.use_oci_target_update:
            return
            
        if self.oci_update_interval == 0:
            return 
        
        if self.num_updates % self.oci_update_interval != 0:
            return

        if self.buffer.num_in_buffer < self.min_buffer_size_before_update:
            return

        for _ in range(self.oci_batch_num):
            oci_batch_size = self.oci_batch_size
            data = self.buffer.sample(oci_batch_size)
            data['target'] = self._get_best_goal_values(data['x'], action_values=True)
            self.behaviour_learner.oci_update(data)

    def update(self, s: Any, a, sp: Any, r, gamma, info=None, terminal: bool = False):
        self.num_updates += 1
        self.num_steps_in_ep += 1

        goal_init = self.goal_initiation_func(s)
        goal_terms = self.goal_termination_func(s, a, sp)

        self._log_model_error()

        if r == 10000:
            self.num_term += 1
            if globals.aux_config.get('show_progress'):
                print(f'terminated! num_term: {self.num_term} num_steps: {self.num_steps_in_ep}')
                # print(f'goal_baseline: {self.goal_estimate_learner.goal_baseline}\ngoal_values: {self.goal_value_learner.goal_values}')
                # print(f'tau: {self.tau}')

                print(f'goal buffer sizes: {[buffer.num_in_buffer for buffer in self.goal_buffers]}')
                print(f'Num in goal estimate buffer: {self.goal_estimate_buffer.num_in_buffer}')
                pass

            globals.collector.collect('num_steps_in_ep', self.num_steps_in_ep)
            self.num_steps_in_ep = 0

        # Exploration bonus:
        # self.tau += 1
        self.tau[np.where(goal_terms == True)] = 0

        if info is not None and info['reset']:
            pass
        else:
            self._add_to_buffer(s, a, sp, r, gamma, terminal, goal_init, goal_terms)

        if self.num_updates > self.prefill_buffer_time:
            if self.learn_model_mode == 'only':
                self._state_to_goal_estimate_update()
                pass
            else:
                pass
                
                if self.learn_model_mode == 'online':
                    self._state_to_goal_estimate_update()
                
                # self._state_to_goal_estimate_update()
                if not self.load_pretrain_goal_values:
                    self._goal_estimate_update()
                    self._goal_value_update() 
                    
                if self.use_oci_target_update:
                    self._oci_target_update()
                else:
                    self._behaviour_update()
                    self._oci(sp)
    
        # Logging
        self.cumulative_reward += r
        def log():
            globals.collector.collect('reward_rate', np.copy(self.cumulative_reward) / globals.blackboard['step_logging_interval'])
            self.cumulative_reward = 0
        run_if_should_log(log)

        if globals.collector.has_key('step_goal_gamma_map'):
            if globals.blackboard['num_steps_passed'] % 10000 == 0:
                # # Calculating the value at each state approximately
                resolution = 40
                num_goals = self.num_goals
                last_q_map = np.zeros((resolution, resolution, 5))
                last_goal_q_map = np.zeros((num_goals, resolution, resolution, 5))
                last_reward_map = np.zeros((num_goals, resolution, resolution, 5))
                last_gamma_map = np.zeros((num_goals, resolution, resolution, 5))
                for r, y in enumerate(np.linspace(0, 1, resolution)):
                    for c, x in enumerate(np.linspace(0, 1, resolution)):
                            for g in range(num_goals):
                                # goal_s = np.append([x, y, 0.0, 0.0], np.array(agent.goals[g]))
                                # action_value, reward, gamma = agent.goal_learner.get_goal_outputs(goal_s)
                                # SWITCHING OVER TO 1 NN PER GOAL
                                action_value, reward, gamma_map = self.goal_learners[g].get_goal_outputs(np.array([x, y, 0.0, 0.0]))
                                last_goal_q_map[g, r, c] = action_value
                                last_reward_map[g, r, c] = reward
                                last_gamma_map[g, r, c] = gamma_map
                                pass

                globals.collector.collect('step_goal_gamma_map', np.copy(last_gamma_map))
    
        if not terminal:
            ap = self.selectAction(sp)
        else:
            ap = None

        return ap

    def agent_end(self, s, a, r, gamma, info=None):
        self.update(s, a, s, r, 0, info=info, terminal=True)

        # self.behaviour_learner.episode_end()
        # self.option_model.episode_end()

    def experiment_end(self):
        # Things to log after the final logging
        save_goal_learner_name = parse_param(self.params, 'save_state_to_goal_estimate_name', lambda p: isinstance(p, str) or p is None, optional=True, default=None)
    
        if save_goal_learner_name is not None:
            cloudpickle.dump(self.goal_learners, open(f'./src/environments/data/pinball/{save_goal_learner_name}_goal_learner.pkl', 'wb'))
            cloudpickle.dump(self.goal_buffers, open(f'./src/environments/data/pinball/{save_goal_learner_name}_goal_buffer.pkl', 'wb'))
            cloudpickle.dump(self.goal_estimate_buffer, open(f'./src/environments/data/pinball/{save_goal_learner_name}_goal_estimate_buffer.pkl', 'wb'))

        # Saving the agent goal learners
        save_behaviour_name = parse_param(self.params, 'save_behaviour_name', lambda p: isinstance(p, str) or p is None, optional=True, default=None)
        if save_behaviour_name:
            cloudpickle.dump(self, open(f'./src/environments/data/pinball/{save_behaviour_name}_agent.pkl', 'wb'))

        if self.save_buffer_name is not None:
             cloudpickle.dump(self.buffer, open(f'./src/environments/data/pinball/{self.save_buffer_name}_buffer.pkl', 'wb'))

        self._log_model_error_heatmap()

        def get_goal_outputs(s, g):
            action_value, reward, gamma = self.goal_learners[g].get_goal_outputs(s)
            return np.vstack([action_value, reward, gamma])

        RESOLUTION = 40
        last_goal_q_map = np.zeros((self.num_goals, RESOLUTION, RESOLUTION, self.num_actions))
        last_reward_map = np.zeros((self.num_goals, RESOLUTION, RESOLUTION, self.num_actions))
        last_gamma_map = np.zeros((self.num_goals, RESOLUTION, RESOLUTION, self.num_actions))
        for g in range(self.num_goals):
            goal_action_value = get_last_pinball_action_value_map(3, partial(get_goal_outputs, g=g), resolution=RESOLUTION)
            last_goal_q_map[g] = goal_action_value[0]
            last_reward_map[g] = goal_action_value[1]
            last_gamma_map[g] = goal_action_value[2]

        globals.collector.collect('goal_q_map', last_goal_q_map)
        globals.collector.collect('goal_r_map', last_reward_map)
        globals.collector.collect('goal_gamma_map', last_gamma_map)

        q_map = get_last_pinball_action_value_map(1, self.behaviour_learner.get_action_values)
        globals.collector.collect('q_map', q_map[0])

class GoalEstimates:
    def __init__(self, num_goals):
        self.num_goals = num_goals

        # initializing weights
        self.r = np.zeros((self.num_goals, self.num_goals))
        self.gamma = np.zeros((self.num_goals, self.num_goals))
        self.goal_baseline = np.zeros(self.num_goals)
        self.goal_init = np.zeros((self.num_goals, self.num_goals))

    def batch_update(self, data, alpha):
        # TODO [2022-04-25] Check out if we can vectorize this
        batch_goal_init = data['goal_inits']
        batch_goal_term = data['goal_terms']
        batch_goal_r = data['goal_rs']
        batch_goal_gamma = data['goal_gammas']
        batch_option_pi_x = data['option_pi_x']
        batch_action_values = data['action_values']

        batch_size = batch_goal_init.shape[0]
        for i in range(batch_size):
            goal_term = batch_goal_term[i]
            option_pi_x = batch_option_pi_x[i]
            goal_r = batch_goal_r[i]
            gamma = batch_goal_gamma[i]
            action_values = batch_action_values[i]
            goal_init = batch_goal_init[i]

            for g in range(self.num_goals):
                if goal_term[g] == True: 
                    self.r[g] += alpha * (np.sum(option_pi_x * goal_r, axis=1) - self.r[g])
                    self.gamma[g] += alpha * (np.sum(option_pi_x * gamma, axis=1)- self.gamma[g])
                    self.goal_baseline[g] += alpha * (np.max(action_values) - self.goal_baseline[g])
                    self.goal_init[g] += alpha * (goal_init - self.goal_init[g])

        def log():
            globals.collector.collect('goal_r', np.copy(self.r))
            globals.collector.collect('goal_gamma', np.copy(self.gamma))
            globals.collector.collect('goal_baseline', np.copy(self.goal_baseline))
            # print(self.goal_baseline)
            # print(self.r[6])
            globals.collector.collect('goal_init', np.copy(self.goal_init))
        run_if_should_log(log)


    def update(self, x, option_pi_x, r_s, gamma_s, alpha, r, xp,  on_goal, x_action_values, goal_init):
        for g in range(self.num_goals):
            if on_goal[g] == True: 
                # if g == 12:
                #     # print(goal_init)
                #     print(x_action_values)
                self.r[g] += alpha * (np.sum(option_pi_x * r_s, axis=1) - self.r[g])
                self.gamma[g] += alpha * (np.sum(option_pi_x * gamma_s, axis=1)- self.gamma[g])
                self.goal_baseline[g] += alpha * (np.max(x_action_values) - self.goal_baseline[g])
                self.goal_init[g] += alpha * (goal_init - self.goal_init[g])

        def log():
            globals.collector.collect('goal_r', np.copy(self.r))
            globals.collector.collect('goal_gamma', np.copy(self.gamma))
            globals.collector.collect('goal_baseline', np.copy(self.goal_baseline))
            globals.collector.collect('goal_init', np.copy(self.goal_init))
        run_if_should_log(log)

class GoalValueLearner:
    def __init__(self, num_goals, terminal_goal_index):
        self.num_goals = num_goals
        self.terminal_goal_index = terminal_goal_index
        
        # Initializing goal valuester
        self.goal_values = np.zeros(self.num_goals)
    
    def update(self, goal_gamma, reward_goals, goal_bonus, goal_init, goal_baseline):
        num_planning_steps = 1
        
        # print(f'goal_gammas: {goal_gamma}')
        for _ in range(num_planning_steps):
            # Just doing value iteration for now 

            # Can we possibly vectorize this?
            for g in range(self.num_goals):
                if g == self.terminal_goal_index:
                    self.goal_values[g] = goal_baseline[g]
                else:
                    valid_goals = np.nonzero(goal_init[g] * goal_gamma[g])[0]
                    if len(valid_goals) > 0:
                        returns = reward_goals[g][valid_goals] + goal_gamma[g][valid_goals] * (self.goal_values[valid_goals] + goal_bonus[valid_goals])
                        self.goal_values[g] = np.max(returns)
        
        def log():
            globals.collector.collect('goal_values', np.copy(self.goal_values)) 
        run_if_should_log(log)