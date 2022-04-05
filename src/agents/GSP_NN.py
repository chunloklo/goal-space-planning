
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
from .components.QLearner_NN import QLearner_NN

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

        self.goal_features = 2
        # self.goal_learner = GoalLearner_NN((4 + self.goal_features,), self.num_actions, self.step_size, 0.1)

        self.goal_learners: List[GoalLearner_NN] = []
        for _ in range(self.num_goals):
            self.goal_learners.append(GoalLearner_NN((4,), self.num_actions, self.step_size, 0.1))

        # self.goal_learner = GoalLearnerIndividual_NN((4,), self.num_actions, self.step_size, 0.1, self.num_goals)

        self.buffer_size = 1000000
        self.min_buffer_size_before_update = 1000
        self.buffer = Buffer(self.buffer_size, {'x': (4,), 'a': (), 'xp': (4,), 'r': (), 'gamma': (), 'goal_inits': (self.num_goals, ), 'goal_terms': (self.num_goals, )}, self.random.randint(0,2**31), type_map={'a': np.int64})
        self.goal_estimate_buffer = Buffer(self.buffer_size, {'x': (4,), 'a': (), 'xp': (4,), 'r': (), 'gamma': (), 'goal_inits': (self.num_goals, ), 'goal_terms': (self.num_goals, )}, self.random.randint(0,2**31))
        self.goal_buffer = Buffer(self.buffer_size * self.num_goals, {'x': (4 + 2,), 'a': (), 'xp': (4 + 2,), 'r': (), 'gamma': (), 'goal_policy_cumulant': (), 'goal_discount': ()}, self.random.randint(0,2**31), type_map={'a': np.int64})

        self.goal_buffers = []
        for _ in range(self.num_goals):
            self.goal_buffers.append(Buffer(self.buffer_size, {'x': (4,), 'a': (), 'xp': (4,), 'r': (), 'gamma': (), 'goal_policy_cumulant': (), 'goal_discount': ()}, self.random.randint(0,2**31), type_map={'a': np.int64}))
        
        self.goal_estimate_learner = GoalEstimates(self.num_goals)
        self.num_steps_in_ep = 0

        self.OCI_update_interval = param_utils.parse_param(params, 'OCI_update_interval', lambda p : isinstance(p, int) and p >= 0) # Number of update steps between each OCI update
        self.use_OCG = param_utils.parse_param(params, 'use_OCG', lambda p : isinstance(p, bool), optional=True, default=False) # Whether to use OCG or not
        self.use_goal_values = param_utils.parse_param(params, 'use_goal_values', lambda p : isinstance(p, bool), optional=True, default=False) # Whether to use the baseline or goal values for OCI
        self.polyak_stepsize = param_utils.parse_param(params, 'polyak_stepsize', lambda p : isinstance(p, float) and p >= 0)
        self.learn_model_only = param_utils.parse_param(params, 'learn_model_only', lambda p : isinstance(p, bool), optional=True, default=False)
        self.goal_value_learner = GoalValueLearner(self.num_goals)
        self.use_pretrained_behavior = param_utils.parse_param(params, 'use_pretrained_behavior', lambda p : isinstance(p, bool), optional=True, default=False)
        self.use_exploration_bonus = param_utils.parse_param(params, 'use_exploration_bonus', lambda p : isinstance(p, bool), optional=True, default=False)

        if self.use_pretrained_behavior:
            agent = pickle.load(open('src/environments/data/pinball/gsp_agent.pkl', 'rb'))
            # self.goal_buffer = agent.goal_buffer
            self.behaviour_learner = agent.behaviour_learner
            self.buffer = agent.buffer
            self.goal_learners = agent.goal_learners
            self.goal_estimate_learner = agent.goal_estimate_learner
            self.goal_buffers = agent.goal_buffers
            self.goal_value_learner = agent.goal_value_learner

            # self.behaviour_learner 
            # print('using pretrained behavior')

        self.use_pretrained_model = param_utils.parse_param(params, 'use_pretrained_model', lambda p : isinstance(p, bool), optional=True, default=False)
        if self.use_pretrained_model:
            self.goal_learners = pickle.load(open('./src/environments/data/pinball/goal_learner.pkl', 'rb'))
                    # cloudpickle.dump(agent.goal_learner, open('./src/environments/data/pinball/goal_learner.pkl', 'wb'))
            self.goal_buffers = pickle.load(open('./src/environments/data/pinball/goal_buffer.pkl', 'rb'))
            
        self.cumulative_reward = 0
        self.num_term = 0
        self.num_updates = 0
        self.tau = np.full(self.num_goals, 13000)

    def FA(self):
        return "Neural Network"

    def __str__(self):
        return "GSP_NN"

    def get_policy(self, s) -> npt.ArrayLike:
        
        action_values = self.behaviour_learner.get_action_values(s)
        if self.use_OCG:
            x_goal_init = self.goal_initiation_func(s)
            x_valid_goals = np.where(x_goal_init == True)[0]

            # tiled_x = np.tile(s, (x_valid_goals.shape[0], 1))
            # goal_x = np.append(tiled_x, x_valid_goals, axis=1)

            # _, goal_rewards, goal_discounts = self.goal_learner.get_goal_outputs(goal_x)

            goal_rewards = []
            goal_discounts = []
            for g in x_valid_goals:
                _, reward, discount = self.goal_learners[g].get_goal_outputs(np.array(s))
                goal_rewards.append(reward)
                goal_discounts.append(discount)

            goal_rewards = np.vstack(goal_rewards)
            goal_discounts = np.vstack(goal_discounts)

            if self.use_exploration_bonus:
                bonus = np.where(self.tau > 12000, 1000, 0)
            else:
                bonus = 0
            if self.use_goal_values:
                goal_value_with_bonus = np.copy(self.goal_value_learner.goal_values) + bonus
            else:
                goal_value_with_bonus = np.copy(self.goal_estimate_learner.goal_baseline) + bonus
            goal_discounts = np.clip(goal_discounts, 0, 1)
                    
            # TODO: I'm not entirely sure whether the leaving initiation problem is fixed or not.
            # It probably isn't, but we can maybe hope that FA is doing a good job?
            valid_goal_values = goal_value_with_bonus[np.where(x_goal_init == True)]
            goal_target = np.max(goal_rewards + goal_discounts * valid_goal_values[:, np.newaxis], axis=0)

            max_action_values = np.maximum(goal_target, action_values)
            a = np.argmax(max_action_values)
        else:
            # epsilon greedy
            a = np.argmax(action_values)
            
        probs = np.zeros(self.num_actions)
        probs += self.epsilon / (self.num_actions)
        probs[a] += 1 - self.epsilon
        return probs

    # public method for rlglue
    def selectAction(self, s: Any) -> int:
        s = np.array(s)
        a = self.random.choice(self.num_actions, p = self.get_policy(s))
        return a
    
    def _add_to_buffer(self, s, a, sp, r, gamma, terminal, goal_inits, goal_terms):
        self.buffer.update({'x': s, 'a': a, 'xp': sp, 'r': r, 'gamma': gamma, 'goal_inits': goal_inits, 'goal_terms': goal_terms})

        if np.any(goal_terms):
            self.goal_estimate_buffer.update({'x': s, 'a': a, 'xp': sp, 'r': r, 'gamma': gamma, 'goal_inits': goal_inits, 'goal_terms': goal_terms})
        
        sp_goal_init = self.goal_initiation_func(sp)

        goal_discount = np.empty(goal_terms.shape)
        goal_discount[goal_terms == True] = 0
        goal_discount[goal_terms == False] = gamma

        goal_policy_cumulant = np.empty(goal_terms.shape)
        goal_policy_cumulant[goal_terms == True] = gamma
        goal_policy_cumulant[goal_terms == False] = 0

        zeros = np.zeros(goal_terms.shape)

        for g in range(self.num_goals):
            # Don't learn with experience that takes you "outside" of the goal initiation zone.
            if goal_inits[g]:
                if goal_terms[g] or sp_goal_init[g] != False:
                    self.goal_buffers[g].update({'x': s, 'a': a, 'xp': sp, 'r': r, 'gamma': gamma, 'goal_policy_cumulant': goal_policy_cumulant[g], 'goal_discount': goal_discount[g]})
                else:
                    self.goal_buffers[g].update({'x': s, 'a': a, 'xp': sp, 'r': r, 'gamma': gamma, 'goal_policy_cumulant': zeros[g], 'goal_discount': goal_discount[g]})
                

    def _direct_rl_update(self):
        if self.buffer.num_in_buffer >= self.min_buffer_size_before_update:
            # Updating behavior
            data = self.buffer.sample(self.batch_size)
            self.behaviour_learner.update(data, polyak_stepsize=self.polyak_stepsize)

    def _state_to_goal_estimate_update(self):
        # if self.goal_buffer.num_in_buffer >= self.min_buffer_size_before_update:
        #     num_batches = 1
        #     for _ in range(num_batches):
        #         data = self.goal_buffer.sample(64)
                # self.goal_learner.update(data, polyak_stepsize=0.0001)

        # for g in range(self.num_goals):
        for g in [7]:
            if self.goal_buffers[g].num_in_buffer >= self.min_buffer_size_before_update:
                # print('updating')
                # print('updating')
                data = self.goal_buffers[g].sample(16)
                self.goal_learners[g].update(data, polyak_stepsize=0.001)            

        ####### Individual
        # goal_learner_data = []
        # all_learn = True

        # for g in range(self.num_goals):
        #     if self.goal_buffers[g].num_in_buffer >= self.min_buffer_size_before_update:
        #         # print('updating')
        #         goal_learner_data.append(self.goal_buffers[g].sample(32))
        #     else:
        #         all_learn = False

        #     if not all_learn:
        #         break
        # if all_learn:
        #     # print('kearning')
        #     self.goal_learner.update(goal_learner_data, polyak_stepsize=0.0005)            


    def _goal_estimate_update(self):
        batch_size = 1
        if self.goal_estimate_buffer.num_in_buffer < batch_size:
            return 
        # print('updating goal estiamtes')
        data = self.goal_estimate_buffer.sample(batch_size)
        
        sps = data['xp']

        # goal_learner_data['goal_inits'] = []
        # goal_learner_data['goal_terms'] = []
        data['goal_rs'] = []
        data['goal_gammas'] = []
        data['option_pi_x'] = []
        data['action_values'] = []

        for i in range(batch_size):
            sp = sps[i]

            action_value = np.empty((self.num_goals, self.num_actions))
            goal_r = np.empty((self.num_goals, self.num_actions))
            goal_gammas = np.empty((self.num_goals, self.num_actions))

            for g in range(self.num_goals):
                action_value[g], goal_r[g], goal_gammas[g] = self.goal_learners[g].get_goal_outputs(np.array(sp))

            # print(goal_r)

            goal_gammas = np.clip(goal_gammas, 0, 1)

            goal_policies = np.zeros((self.num_goals, self.num_actions))
            goal_policies[np.arange(self.num_goals), np.argmax(action_value, axis=1)] = 1 
            current_val =  self.behaviour_learner.get_target_action_values(np.array(sp))

            data['goal_rs'].append(goal_r)
            data['goal_gammas'].append(goal_gammas)
            data['option_pi_x'].append(goal_policies)
            data['action_values'].append(current_val)

            # if data['goal_terms'][0][6] == True:
                # print('test')
                # print(s)
                # print(self.goal_learners[7].get_goal_outputs(np.array(s)))
                # print(goal_r[])
                # print(goal_r[7])
                # print(goal_policies[7])
                # print((np.sum(goal_policies * goal_r, axis=1)))

        self.goal_estimate_learner.batch_update(data, 0.01)

    def _goal_value_update(self):
        if self.use_exploration_bonus:
                bonus = np.where(self.tau > 12000, 1000, 0)
        else:
            bonus = 0
        self.goal_value_learner.update(self.goal_estimate_learner.gamma, self.goal_estimate_learner.r, bonus, self.goal_estimate_learner.goal_init, self.goal_estimate_learner.goal_baseline)

    def _OCI(self, sp):
        ############################ OCI
        if self.buffer.num_in_buffer >= self.min_buffer_size_before_update:
            
            # OCI. Resampling here
            ######### OCI. Resampling here
            OCI_batch_size = self.batch_size
            data = self.buffer.sample(OCI_batch_size)

            if self.use_exploration_bonus:
                bonus = np.where(self.tau > 12000, 1000, 0)
            else:
                bonus = 0

            if self.use_goal_values:
                goal_value_with_bonus = np.copy(self.goal_value_learner.goal_values) + bonus
            else:
                goal_value_with_bonus = np.copy(self.goal_estimate_learner.goal_baseline) + bonus

            data['target'] = np.empty((OCI_batch_size, self.num_actions))

            # Grabbing all goal rewards and discounts
            for i in range(OCI_batch_size):
                x = data['x'][i]
                x_goal_init = self.goal_initiation_func(x)
                valid_goals = np.where(x_goal_init == True)[0]

                if len(valid_goals) <= 0:
                    continue

                x_rewards = np.empty((len(valid_goals), self.num_actions))
                x_discounts = np.empty((len(valid_goals), self.num_actions))

                for i_g, g in enumerate(valid_goals):
                    _, reward, discount = self.goal_learners[g].get_goal_outputs(x)
                    discount = np.clip(discount, 0, 1)
                    x_rewards[i_g] = reward
                    x_discounts[i_g] = discount

                valid_goal_values = goal_value_with_bonus[valid_goals]
                goal_target = np.max(x_rewards + x_discounts * valid_goal_values[:, np.newaxis], axis=0)
                data['target'][i] = goal_target

            self.behaviour_learner.OCI_update(data, polyak_stepsize=self.polyak_stepsize)


            ########## TARGET UPDATE RATHER THAN BASE
            ########## OCI. Resampling here
            # OCI_batch_size = self.batch_size
            # data = self.buffer.sample(OCI_batch_size)

            # goal_targets = []
            # # bonus = np.where(self.tau > 12000, 1000, 0)
            # bonus = 0
            # if self.use_goal_values:
            #     goal_value_with_bonus = np.copy(self.goal_value_learner.goal_values) + bonus
            # else:
            #     goal_value_with_bonus = np.copy(self.goal_estimate_learner.goal_baseline) + bonus

            # data['xp_goal_target'] = np.empty((OCI_batch_size, self.num_actions))

            # # Grabbing all goal rewards and discounts
            # for i in range(OCI_batch_size):
            #     xp = data['xp'][i]
            #     xp_goal_init = self.goal_initiation_func(xp)
            #     valid_goals = np.where(xp_goal_init == True)[0]

            #     xp_rewards = np.empty((len(valid_goals), self.num_actions))
            #     xp_discounts = np.empty((len(valid_goals), self.num_actions))

            #     for i_g, g in enumerate(valid_goals):
            #         _, reward, discount = self.goal_learners[g].get_goal_outputs(xp)
            #         discount = np.clip(discount, 0, 1)
            #         xp_rewards[i_g] = reward
            #         xp_discounts[i_g] = discount

            #     valid_goal_values = goal_value_with_bonus[valid_goals]
            #     goal_target = np.max(xp_rewards + xp_discounts * valid_goal_values[:, np.newaxis], axis=0)
            #     data['xp_goal_target'][i] = goal_target

            # self.behaviour_learner.OCI_update(data, polyak_stepsize=self.polyak_stepsize)


    def update(self, s: Any, a, sp: Any, r, gamma, terminal: bool = False):
        # print(s[2:])
        self.num_updates += 1
        self.num_steps_in_ep += 1


        goal_init = self.goal_initiation_func(s)
        goal_terms = self.goal_termination_func(sp)

        if r == 10000:
            self.num_term += 1
            # print(s, sp)
            # print(goal_init)
            # print(goal_terms)
            if globals.aux_config.get('show_progress'):
                print(f'terminated! num_term: {self.num_term} num_steps: {self.num_steps_in_ep}')

                print(self.goal_estimate_learner.goal_baseline)

            # print(f'goal_baseline: {self.goal_estimate_learner.goal_baseline}\ngoal_values: {self.goal_value_learner.goal_values}')
            globals.collector.collect('num_steps_in_ep', self.num_steps_in_ep)
            self.num_steps_in_ep = 0

        # Exploration bonus:
        # self.tau += 1
        self.tau[np.where(goal_terms == True)] = 0

        # print(self.tau)

        self._add_to_buffer(s, a, sp, r, gamma, terminal, goal_init, goal_terms)

        if self.learn_model_only:
            # print('kesdf')
            self._state_to_goal_estimate_update()
        else:
            self._direct_rl_update()
            # self._state_to_goal_estimate_update()
            self._goal_estimate_update()
            self._goal_value_update() # Taking out  goal value for now
            if self.OCI_update_interval > 0:
                if self.num_updates % self.OCI_update_interval == 0:
                    self._OCI(sp)
    
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


class GoalLearnerIndividual_NN():
    def __init__(self, state_shape, num_actions: int, step_size: float, epsilon: float, num_goals: int):
        self.num_actions: int = num_actions
        self.epsilon = epsilon
        self.num_goals = num_goals

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

            # q_mlp = hk.Sequential([
            #     hk.Linear(128), jax.nn.relu,
            #     hk.Linear(128), jax.nn.relu,
            #     hk.Linear(self.num_actions),
            # ])

            # gamma_head = hk.Linear(self.num_actions)
            # value_head = hk.Linear(self.num_actions)

            # return gamma_head(q_mlp(states)), value_head(q_mlp(states))

        self.f_qfunc = hk.without_apply_rng(hk.transform(q_function))

        self.f_opt = optax.adam(step_size)

        def _take_action_index(data, action):
            return jnp.take_along_axis(data, jnp.expand_dims(action, axis=1), axis=1).squeeze()
    
        def _loss(params: List[hk.Params], target_params: List[hk.Params], data):
            
            policy_loss = 0
            reward_loss = 0
            for g in range(self.num_goals):
                r = data[g]['r']
                x = data[g]['x']
                a = data[g]['a']
                xp = data[g]['xp']
                # gamma = data['gamma']
                goal_policy_cumulant = data[g]['goal_policy_cumulant']
                goal_discount = data[g]['goal_discount']

                policy_pred, reward_pred, gamma_pred = self.f_get_goal_output(params[g], x)
                # Getting values for specific actions
                policy_pred = _take_action_index(policy_pred, a)
                reward_pred = _take_action_index(reward_pred, a)
                gamma_pred = _take_action_index(gamma_pred, a)
                
                xp_policy_pred, xp_reward_pred, xp_gamma_pred = jax.lax.stop_gradient(self.f_get_goal_output(target_params[g], xp))
                xp_ap = jnp.argmax(xp_policy_pred, axis=1)

                # ESarsa version rather than Q version
                policy = jax.nn.one_hot(xp_ap, self.num_actions) * (1 - self.epsilon) + (self.epsilon / self.num_actions)
                policy_target = goal_policy_cumulant + goal_discount * jnp.average(xp_policy_pred, axis=1, weights=policy)
                reward_target = r + goal_discount * jnp.average(xp_reward_pred, axis=1, weights=policy)

                # policy_target = goal_policy_cumulant + goal_discount * _take_action_index(xp_policy_pred, xp_ap)
                # reward_target = r + goal_discount * _take_action_index(xp_reward_pred, xp_ap)

                policy_loss += jnp.mean(jnp.square(policy_target - policy_pred))
                reward_loss += jnp.mean(jnp.square(reward_target - reward_pred))

            return policy_loss, (jax.lax.stop_gradient(policy_loss), jax.lax.stop_gradient(reward_loss))

            return policy_loss + reward_loss, (jax.lax.stop_gradient(policy_loss), jax.lax.stop_gradient(reward_loss))

        def _get_goal_output(params: hk.Params, x: Any):
            policy_output, v_output = self.f_qfunc.apply(params, x)
            # Since the problem is shortest path and we're trying to maximize the reward anyways, the signals are essentially identical.
            gamma_output = policy_output
            return policy_output, v_output, gamma_output

        def _update(params: List[hk.Params], target_params: List[hk.Params], opt_state, data):
            grads, reward_loss = jax.grad(_loss, has_aux=True)(params, target_params, data)
            updates, opt_state = self.f_opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, reward_loss
            
        self.f_get_goal_output = jax.jit(_get_goal_output)
        self.f_update = jax.jit(_update)
        # self.f_update = _update

        # Initialization
        dummy_state = jnp.zeros(state_shape)
        self.params = []
        rng = jax.random.PRNGKey(42)
        for i in range(self.num_goals):
            self.params.append(self.f_qfunc.init(rng, dummy_state))
        self.opt_state = self.f_opt.init(self.params)

        # target params for the network
        self.target_params = copy.deepcopy(self.params)

    def get_goal_outputs(self, x: npt.ArrayLike) -> np.ndarray:
        return self.f_get_goal_output(self.params, x)

    def update(self, data, polyak_stepsize:float=0.005):
        self.params, self.opt_state, (policy_loss, reward_loss) = self.f_update(self.params, self.target_params, self.opt_state, data)
        
        # def log():
        #     globals.collector.collect('reward_loss', reward_loss)
        #     globals.collector.collect('policy_loss', policy_loss)
        # run_if_should_log(log)

        self.target_params = optax.incremental_update(self.params, self.target_params, polyak_stepsize)
        return self.params

class GoalLearner_NN():
    def __init__(self, state_shape, num_actions: int, step_size: float, epsilon: float):
        self.num_actions: int = num_actions
        self.epsilon = epsilon

        # self.num_outputs: int = num_actions * 3 # adding reward and gamma output
        # self.policy_slice = slice(0, num_actions) # slice for the greedy goal reaching policy
        # self.reward_slice = slice(num_actions, num_actions * 2) # slice for the reward following greedy policy till termination
        # self.gamma_slice = slice(num_actions * 2, num_actions * 3)# slice for the discount following greedy policy till termination
        
        # Initializing jax functions
        def q_function(states):
            q_mlp = hk.Sequential([
                hk.Linear(64), jax.nn.relu,
                hk.Linear(64), jax.nn.relu,
                hk.Linear(64), jax.nn.relu,
                hk.Linear(64), jax.nn.relu,
                hk.Linear(self.num_actions),
            ])

            v_mlp = hk.Sequential([
                hk.Linear(64), jax.nn.relu,
                hk.Linear(64), jax.nn.relu,
                hk.Linear(64), jax.nn.relu,
                hk.Linear(64), jax.nn.relu,
                hk.Linear(self.num_actions),
            ])

            return q_mlp(states), v_mlp(states)

            # q_mlp = hk.Sequential([
            #     hk.Linear(128), jax.nn.relu,
            #     hk.Linear(128), jax.nn.relu,
            #     hk.Linear(self.num_actions),
            # ])

            # gamma_head = hk.Linear(self.num_actions)
            # value_head = hk.Linear(self.num_actions)

            # return gamma_head(q_mlp(states)), value_head(q_mlp(states))

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

            # ESarsa version rather than Q version
            # policy = jax.nn.one_hot(xp_ap, self.num_actions) * (1 - self.epsilon) + (self.epsilon / self.num_actions)
            # policy_target = goal_policy_cumulant + goal_discount * jnp.average(xp_policy_pred, axis=1, weights=policy)
            # reward_target = r + goal_discount * jnp.average(xp_reward_pred, axis=1, weights=policy)

            policy_target = goal_policy_cumulant + goal_discount * _take_action_index(xp_policy_pred, xp_ap)
            reward_target = r + goal_discount * _take_action_index(xp_reward_pred, xp_ap)

            policy_loss = jnp.mean(jnp.square(policy_target - policy_pred))
            reward_loss = jnp.mean(jnp.square(reward_target - reward_pred))

            # return policy_loss, (jax.lax.stop_gradient(policy_loss), jax.lax.stop_gradient(reward_loss))

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
        run_if_should_log(log)

        self.target_params = optax.incremental_update(self.params, self.target_params, polyak_stepsize)
        # return self.params
    
class GoalEstimates:
    def __init__(self, num_goals):
        self.num_goals = num_goals

        # initializing weights
        self.r = np.zeros((self.num_goals, self.num_goals))
        self.gamma = np.zeros((self.num_goals, self.num_goals))
        self.goal_baseline = np.zeros(self.num_goals)
        self.goal_init = np.zeros((self.num_goals, self.num_goals))

    def batch_update(self, data, alpha):
        batch_goal_init = data['goal_inits']
        batch_goal_term = data['goal_terms']
        batch_goal_r = data['goal_rs']
        batch_goal_gamma = data['goal_gammas']
        batch_option_pi_x = data['option_pi_x']
        batch_action_values = data['action_values']
        # print(batch_action_values)

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

                    # if g == 6:
                    #     print('start')
                    #     print(goal_r[7])
                    #     print(option_pi_x[7])
                    #     print((np.sum(option_pi_x * goal_r, axis=1)))
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
                valid_goals = np.nonzero(goal_init[g])[0]
                if g == 3:
                    valid_goals = []

                if len(valid_goals) > 0:
                    # if g == 0:
                    #     print(reward_goals[g])
                    #     print(goal_gamma[g])
                    #     print(self.goal_values)
                    # print(reward_goals[g])
                    returns = reward_goals[g] + goal_gamma[g] * (self.goal_values + goal_bonus)

                    # if any(np.isnan(returns)):

                    old_goal_values = np.copy(self.goal_values)
                        
                    self.goal_values[g] = np.max(returns[valid_goals])


                    # if not np.isfinite(self.goal_values[g]):
                    #     print(g)
                    #     print(valid_goals)
                    #     print(reward_goals[g])
                    #     print(goal_gamma[g])
                    #     print(old_goal_values)
                    #     print(self.goal_values)
                    #     print(returns)
                    #     asdas



                    # print(returns[valid_goals[0]], goal_baseline[g])
                else:
                    self.goal_values[g] = goal_baseline[g]




        
        if globals.blackboard['num_steps_passed'] % globals.blackboard['step_logging_interval'] == 0:
            # print(f'values: {self.goal_values}')

            # print(self.goal)
            # print(self.goal_values[3], goal_baseline[3])
            globals.collector.collect('goal_values', np.copy(self.goal_values)) 