
from functools import partial
import numpy as np
from src.utils import rlglue
from src.utils import globals
from src.utils import options, param_utils
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
from .components.EQRC_model import GoalLearner_EQRC_NN
from .components.DQN_model import GoalLearner_DQN_NN
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

        # Initializing goal information
        self.goals = problem.goals
        self.num_goals = self.goals.num_goals
        self.goal_termination_func = self.goals.goal_termination
        self.goal_initiation_func = self.goals.goal_initiation

        params = self.params
        # Controls the number of samples to sample from the buffer when performing an update 
        self.batch_size = param_utils.parse_param(params, 'batch_size', lambda p : isinstance(p, int) and p >= 0.0)
        # Step size for both the behaviour and model learner. These are combined for now
        self.step_size = param_utils.parse_param(params, 'step_size', lambda p : isinstance(p, float) and p >= 0.0) 
        # Epsilon for epsilon-greedy policy
        self.epsilon = param_utils.parse_param(params, 'epsilon', lambda p : isinstance(p, float) and p >= 0.0)
        # Polyak stepsize for DQN and model learning. They are combined for now
        self.polyak_stepsize = param_utils.parse_param(params, 'polyak_stepsize', lambda p : isinstance(p, float) and p >= 0)


        self.OCI_update_interval = param_utils.parse_param(params, 'OCI_update_interval', lambda p : isinstance(p, int) and p >= 0) # Number of update steps between each OCI update
        
        # Optional parameters

        # Whether to use the baseline or goal values for OCI
        self.use_goal_values = param_utils.parse_param(params, 'use_goal_values', lambda p : isinstance(p, bool), optional=True, default=False) 
        # Whether the agent should only learn the model, and not learn the behaviour
        self.learn_model_only = param_utils.parse_param(params, 'learn_model_only', lambda p : isinstance(p, bool), optional=True, default=False)
        # Additional parameter for controlling adam's epsilon parameter for model learning
        self.adam_eps = param_utils.parse_param(params, 'adam_eps', lambda p : isinstance(p, float) and p >= 0, optional=True, default=1e-8)
        # Whether to use goal-based exploration bonus or not
        self.use_exploration_bonus = param_utils.parse_param(params, 'use_exploration_bonus', lambda p : isinstance(p, bool), optional=True, default=False)
        # The amount of time to simply add to the buffer and not learn anything 
        self.prefill_buffer_time = param_utils.parse_param(params, 'prefill_buffer_time', lambda p : isinstance(p, int) and p >= 0, optional=True, default=0)
        # Whether to only train the state to goal models on specific goals.
        # For now, the list is a tuple since lists aren't hashable.
        self.learn_select_goal_models = param_utils.parse_param(params, 'learn_select_goal_models', lambda p : isinstance(p, tuple) or p is None, optional=True, default=None)

        # Name of the file that contains the pretrained behavior that the agent should load. If None, its starts from scratch
        self.pretrained_behavior_name = param_utils.parse_param(params, 'pretrained_behavior_name', lambda p : isinstance(p, str) or p is None, optional=True, default=None)
        # Name for the pre-trained model
        self.pretrained_model_name = param_utils.parse_param(params, 'pretrained_model_name', lambda p : isinstance(p, str) or p is None, optional=True, default=None)
        # Name for the pre-filled buffer
        self.prefill_buffer = param_utils.parse_param(params, 'prefill_buffer_name', lambda p : isinstance(p, str) or p is None, optional=True, default=None)
    
        # Some fixed parameters that might want to get parameterized later
        self.buffer_size = 1000000
        self.min_buffer_size_before_update = 10000
        # Hard coding this for the environment for now
        self.obs_shape = (4, )
        
        self.goal_value_learner = GoalValueLearner(self.num_goals)
        self.behaviour_learner = QLearner_NN(self.obs_shape, 5, self.step_size)

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
        
        self.goal_estimate_learner = GoalEstimates(self.num_goals)
        self.goal_learners = [GoalLearner_EQRC_NN(self.obs_shape, self.num_actions, self.step_size, 0.1, beta=1.0) for _ in range(self.num_goals)]
        # self.goal_learners = [GoalLearner_DQN_NN(self.obs_shape, self.num_actions, self.step_size, 0.1, self.polyak_stepsize, self.adam_eps) for _ in range(self.num_goals)]

        if self.pretrained_behavior_name:
            agent = pickle.load(open('src/environments/data/pinball/gsp_agent.pkl', 'rb'))
            self.behaviour_learner = agent.behaviour_learner
            self.buffer = agent.buffer
            self.goal_learners = agent.goal_learners
            self.goal_estimate_learner = agent.goal_estimate_learner
            self.goal_buffers = agent.goal_buffers
            self.goal_value_learner = agent.goal_value_learner

        # [chunlok 20202-04-15] TODO The specific path of these saved models might need to be changed later to be more general
        if self.pretrained_model_name is not None:
            self.goal_learners = pickle.load(open(f'./src/environments/data/pinball/{self.pretrained_model_name}_goal_learner.pkl', 'rb'))
            self.goal_buffers = pickle.load(open(f'./src/environments/data/pinball/{self.pretrained_model_name}_goal_buffer.pkl', 'rb'))
            self.goal_estimate_buffer = pickle.load(open(f'./src/environments/data/pinball/{self.pretrained_model_name}_goal_estimate_buffer.pkl', 'rb'))
            for g in range(self.num_goals):
                possible.append(self.goal_buffers[g].num_in_buffer)
            asdsa

        self.prefill_buffer = param_utils.parse_param(params, 'use_prefill_buffer', lambda p : isinstance(p, str) or p is None, optional=True, default=None)
        if self.prefill_buffer is not None:
            self.goal_buffers = pickle.load(open(f'./src/environments/data/pinball/{self.prefill_buffer}_goal_buffer.pkl', 'rb'))

            possible = []
            for g in range(self.num_goals):
                possible.append(self.goal_buffers[g].num_in_buffer)
                #     possible.append(True)
                # else:
                #     possible.append(False)
            print(f'Size of each goal buffer: {possible}')

        self.cumulative_reward = 0
        self.num_term = 0
        self.num_updates = 0
        self.tau = np.full(self.num_goals, 13000)
        self.num_steps_in_ep = 0

    def FA(self):
        return "Neural Network"

    def __str__(self):
        return "GSP_NN"

    def get_policy(self, s) -> npt.ArrayLike:

        if self.epsilon == 1.0:
            return np.full(self.num_actions, 1.0 / self.num_actions)
        
        action_values = self.behaviour_learner.get_action_values(s)
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
    
    def _add_to_buffer(self, s, a, sp, r, gamma, terminal, goal_inits, goal_terms):
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
            # If the experience takes you outside the initiation zone, then the policy definitely doesn't want to go that way.
            if goal_inits[g]:
                if goal_terms[g] or sp_goal_init[g] != False:
                    self.goal_buffers[g].update({'x': s, 'a': a, 'xp': sp, 'r': r, 'gamma': gamma, 'goal_policy_cumulant': goal_policy_cumulant[g], 'goal_discount': goal_discount[g]})
                else:
                    self.goal_buffers[g].update({'x': s, 'a': a, 'xp': sp, 'r': r, 'gamma': gamma, 'goal_policy_cumulant': zeros[g], 'goal_discount': zeros[g]})

    def _direct_rl_update(self):
        if self.buffer.num_in_buffer >= self.min_buffer_size_before_update:
            # Updating behavior
            data = self.buffer.sample(self.batch_size)
            self.behaviour_learner.update(data, polyak_stepsize=self.polyak_stepsize)

    def _state_to_goal_estimate_update(self):
        # for g in range(self.num_goals):
        iter = self.learn_select_goal_models if self.learn_select_goal_models is not None else range(self.num_goals)
        for g in iter:
            if self.goal_buffers[g].num_in_buffer >= self.min_buffer_size_before_update:
                batch_num = 1
                for _ in range(batch_num):
                    data = self.goal_buffers[g].sample(self.batch_size, copy=False)

                    # if np.sum(data['goal_policy_cumulant']) > 0:
                    #     index = np.where(data['goal_policy_cumulant'] > 0)[0][0]
                    #     state = data['x'][index]
                    #     sp = data['xp'][index]
                    #     action = data['a'][index]
                    #     print(index, state, action, sp)

                    #     print(f'before {self.goal_learners[g].get_goal_outputs(state)[0][action]}')
                    #     print(self.goal_termination_func(state, action, sp))
                        
                    #     goal_speeds = self.goals.goal_speeds
                    #     speed_radius_squared = self.goals.speed_radius_squared
                    #     speed_close = np.sum(np.power(sp[2:] - goal_speeds, 2), axis=1) <= speed_radius_squared
                    #     print('getting term')
                    #     print(speed_close)

                        # print(np.where(data['goal_policy_cumulant'] > 0))
                        # print('hi')
                    self.goal_learners[g].update(data) 
                    # self.goal_learners[g].update(data) 

                    # if np.sum(data['goal_policy_cumulant']) > 0:
                    #     index = np.where(data['goal_policy_cumulant'] > 0)[0][0]
                    #     state = data['x'][index]
                    #     action = data['a'][index]
                    #     # print(index, state, action)

                    #     print(f'after {self.goal_learners[g].get_goal_outputs(state)[0][action]}')
    
    def _goal_estimate_update(self):
        batch_size = 1
        if self.goal_estimate_buffer.num_in_buffer < batch_size:
            return 
        data = self.goal_estimate_buffer.sample(batch_size)
        
        sps = data['xp']
        data['goal_rs'] = np.zeros((batch_size, self.num_goals, self.num_actions))
        data['goal_gammas'] = np.zeros((batch_size, self.num_goals, self.num_actions))
        data['option_pi_x'] = np.zeros((batch_size, self.num_goals, self.num_actions))
        data['action_values'] = np.zeros((batch_size, self.num_goals, self.num_actions))

        for i in range(batch_size):
            sp = sps[i]

            action_value = np.empty((self.num_goals, self.num_actions))
            goal_r = np.empty((self.num_goals, self.num_actions))
            goal_gammas = np.empty((self.num_goals, self.num_actions))

            for g in range(self.num_goals):
                action_value[g], goal_r[g], goal_gammas[g] = self.goal_learners[g].get_goal_outputs(np.array(sp))

            goal_gammas = np.clip(goal_gammas, 0, 1)

            goal_policies = np.zeros((self.num_goals, self.num_actions))
            goal_policies[np.arange(self.num_goals), np.argmax(action_value, axis=1)] = 1 
            current_val =  self.behaviour_learner.get_target_action_values(np.array(sp))

            data['goal_rs'][i] = goal_r
            data['goal_gammas'][i] = goal_gammas
            data['option_pi_x'][i] = (goal_policies)
            data['action_values'][i] = (current_val)

        self.goal_estimate_learner.batch_update(data, 0.005)

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

    def update(self, s: Any, a, sp: Any, r, gamma, info=None, terminal: bool = False):
        self.num_updates += 1
        self.num_steps_in_ep += 1

        goal_init = self.goal_initiation_func(s)
        goal_terms = self.goal_termination_func(s, a, sp)

        if r == 10000:
            self.num_term += 1
            # print(s, sp)
            # print(goal_init)
            # print(goal_terms)
            if globals.aux_config.get('show_progress'):
                print(f'terminated! num_term: {self.num_term} num_steps: {self.num_steps_in_ep} tau: {self.tau}')
                
                possible = []
                for g in range(self.num_goals):
                    possible.append(self.goal_buffers[g].num_in_buffer)
                    #     possible.append(True)
                    # else:
                    #     possible.append(False)
                print(f'goal buffer enough: {possible}')
                print(f'goal_baseline: {self.goal_estimate_learner.goal_baseline}\ngoal_values: {self.goal_value_learner.goal_values}')
                print(self.goal_estimate_learner.goal_baseline)
                pass

            
            globals.collector.collect('num_steps_in_ep', self.num_steps_in_ep)
            self.num_steps_in_ep = 0

        # Exploration bonus:
        # self.tau += 1
        self.tau[np.where(goal_terms == True)] = 0

        # print(self.tau)

        if info is not None and info['reset']:
            pass
        else:
            self._add_to_buffer(s, a, sp, r, gamma, terminal, goal_init, goal_terms)

        if self.num_updates > self.prefill_buffer_time:
            if self.learn_model_only:
                self._state_to_goal_estimate_update()
                pass
            else:
                self._direct_rl_update()
                # self._state_to_goal_estimate_update()
                self._goal_estimate_update()
                self._goal_value_update() # Taking out  goal value for now
                # if self.OCI_update_interval > 0:
                #     if self.num_updates % self.OCI_update_interval == 0:
                #         self._OCI(sp)
    
        # # Logging
        self.cumulative_reward += r
        def log():
            globals.collector.collect('reward_rate', np.copy(self.cumulative_reward) / globals.blackboard['step_logging_interval'])
            self.cumulative_reward = 0

            # possible = []
            # for g in range(self.num_goals):
            #     possible.append(self.goal_buffers[g].num_in_buffer)
            #     #     possible.append(True)
            #     # else:
            #     #     possible.append(False)
            # print(f'goal buffer enough: {possible}')
            
        run_if_should_log(log)


        # try:
        if globals.blackboard['num_steps_passed'] % 5000 == 0:
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
            # globals.collector.collect('step_goal_q_map', np.copy(last_goal_q_map))
            # globals.collector.collect('step_goal_reward_map', np.copy(last_goal_q_map))
            globals.collector.collect('step_goal_gamma_map', np.copy(last_gamma_map))

        # except KeyError as e:
        #     pass
    
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
        save_goal_learner_name = param_utils.parse_param(self.params, 'save_state_to_goal_estimate_name', lambda p: isinstance(p, str) or p is None, optional=True, default=None)
    
        if save_goal_learner_name is not None:
            cloudpickle.dump(self.goal_learners, open(f'./src/environments/data/pinball/{save_goal_learner_name}_goal_learner.pkl', 'wb'))
            cloudpickle.dump(self.goal_buffers, open(f'./src/environments/data/pinball/{save_goal_learner_name}_goal_buffer.pkl', 'wb'))
            cloudpickle.dump(self.goal_estimate_buffer, open(f'./src/environments/data/pinball/{save_goal_learner_name}_goal_estimate_buffer.pkl', 'wb'))

        # Saving the agent goal learners
        save_behaviour_name = param_utils.parse_param(self.params, 'save_behaviour_name', lambda p: isinstance(p, bool), optional=True, default=False)
        if save_behaviour_name:
            cloudpickle.dump(self, open(f'./src/environments/data/pinball/{save_behaviour_name}_agent.pkl', 'wb'))

        def get_goal_outputs(s, g):
            action_value, reward, gamma = self.goal_learners[g].get_goal_outputs(s)
            return np.vstack([action_value, reward, gamma])

        RESOLUTION = 40
        last_goal_q_map = np.zeros((self.num_goals, RESOLUTION, RESOLUTION, self.num_actions))
        last_reward_map = np.zeros((self.num_goals, RESOLUTION, RESOLUTION, self.num_actions))
        last_gamma_map = np.zeros((self.num_goals, RESOLUTION, RESOLUTION, self.num_actions))
        for g in range(self.num_goals):
            goal_action_value = get_last_pinball_action_value_map(3, partial(get_goal_outputs, g=g))
            last_goal_q_map[g] = goal_action_value[0]
            last_reward_map[g] = goal_action_value[1]
            last_gamma_map[g] = goal_action_value[2]

        globals.collector.collect('goal_q_map', last_goal_q_map)
        globals.collector.collect('goal_r_map', last_reward_map)
        globals.collector.collect('goal_gamma_map', last_gamma_map)
        # print(last_gamma_map)

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