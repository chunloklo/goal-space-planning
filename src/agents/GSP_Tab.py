from typing import Dict, Optional, Union, Tuple
import numpy as np
from numpy import isin
import numpy.typing as npt
from PyExpUtils.utils.random import argmax, choice
import random
from PyFixedReps.Tabular import Tabular
from src.agents.components.learners import ESarsaLambda, QLearner
from src.agents.components.search_control import ActionModelSearchControl_Tabular
from src.utils import rlglue
from src.utils import globals
from src.utils import options, param_utils
from src.agents.components.models import OptionModel_Sutton_Tabular, CombinedModel_ESarsa_Tabular
from src.agents.components.approximators import DictModel
from typing import Dict, Union, Tuple, Any, TYPE_CHECKING
from src.utils.log_utils import run_if_should_log

from src.utils.numpy_utils import create_onehot
# from src.environments.GrazingWorldAdam import get_pretrained_option_model, state_index_to_coord, get_all_transitions

if TYPE_CHECKING:
    # Important for forward reference
    from src.problems.BaseProblem import BaseProblem

class GSP_Tab:
    def __init__(self, problem: 'BaseProblem'):
        self.wrapper_class = rlglue.OneStepWrapper

        self.env = problem.getEnvironment()
        self.representation = problem.get_representation('Tabular')
        self.features = self.representation.features
        self.num_actions = problem.actions
        self.actions = list(range(self.num_actions))
        self.params = problem.params
        self.num_states = self.env.nS
        self.random = np.random.RandomState(problem.seed)

        params = self.params
        self.step_size = param_utils.parse_param(params, 'step_size', lambda p: isinstance(p, float))
        self.epsilon = param_utils.parse_param(params, 'epsilon', lambda p: p >= 0 and p <= 1)
        self.kappa = param_utils.parse_param(params, 'kappa', lambda p: p >=0)

        self.behaviour_learner = QLearner(self.num_states + 1, self.num_actions)
    
        # Get the list of goals
        self.goal_states = np.array(problem.get_goals())

        self.num_goals = len(self.goal_states)

        self.goal_policies = problem.get_learned_goal_policies()

        # State estimate learner
        self.state_estimate_learner = ESARSAStateEstimates(self.num_states + 1, self.num_actions, self.num_goals)

        def on_goal(x):
            return x == self.goal_states
        
        self.goal_func = on_goal
        self.goal_estimate_learner = GoalEstimates(self.num_goals)
        self.goal_value_learner = GoalValueLearner(self.num_goals)

        # Exploration bonus
        self.tau = np.zeros(self.num_goals)
        
        # Creating models for actions and options
        self.history_dict = DictModel()
        self.search_control = ActionModelSearchControl_Tabular(self.random)


        # Overriding if using pretrained model:
        self.use_pretrained_model = param_utils.parse_param(params, 'use_pretrained_model', lambda p: isinstance(p, bool), optional=True, default=False)
        if self.use_pretrained_model:
            self.state_estimate_learner, self.goal_estimate_learner, self.goal_value_learner = problem.get_pretrained_GSP_models()
        # For logging
        self.cumulative_reward = 0
        

    def FA(self):
        return "Tabular"

    def __str__(self):
        return "GSP_Tab"
    
    def get_policy(self, x: int) -> npt.ArrayLike:
        # Implements epsilon greedy here
        probs = np.zeros(self.num_actions)
        probs += self.epsilon / (self.num_actions)
        a = np.argmax(self.behaviour_learner.get_action_values(x))
        probs[a] += 1 - self.epsilon
        return probs

    # Public method for rlglue
    def selectAction(self, s: int) -> Tuple[int, int] :
        x = self.representation.encode(s)
        a = self.random.choice(self.num_actions, p = self.get_policy(x))
        return a

    def update(self, s, a, sp, r, gamma, terminal: bool = False):

        # Feature representation
        x = self.representation.encode(s)
        # Treating the terminal state as an additional state in the tabular setting
        xp = self.representation.encode(sp) if not terminal else self.num_states

        # print(self.representation.encode((8, 1)))
        # print(self.env.valid_grids)
        # print(self.representation.encode_map)


        # Exploration bonus
        if not globals.blackboard['in_exploration_phase']:
            self.tau += 1
        self.tau[xp == self.goal_states] = 0

        self.history_dict.update(x, a, xp, r, gamma)
        self.search_control.update(x, xp)

        # Direct RL Update
        self.behaviour_learner.update(x, a, xp, r, gamma, self.step_size)

        self._update_state_estimates(x, a, xp, r, gamma)
        self._map_goal_to_state_estimates(x, a, xp, r, gamma)
        self._improve_goal_values()
        self._option_constrained_improvement(x, xp)

        if not terminal:
            ap = self.selectAction(sp)
        else:
            ap = None

        self.cumulative_reward += r

        def log(): 
            globals.collector.collect('Q', np.copy(self.behaviour_learner.Q)) 
            globals.collector.collect('reward_rate', np.copy(self.cumulative_reward) / globals.blackboard['step_logging_interval'])
            self.cumulative_reward = 0
        run_if_should_log(log)

        return ap

    def _get_goal_policy_term(self, x: int, g: int):
        if x is self.num_states:
            # Have the option always take a specific action if xp terminates
            action, term = 0, True
        else:
            action = self.goal_policies[g][x]
            term = self.goal_states[g] == x

        return action, term


    def _update_state_estimates(self, x, a, xp, r, gamma):
        option_pi_xp = np.zeros((self.num_goals, self.num_actions))
        x_option_gamma = np.zeros(self.num_goals)
        valid_goal = np.full(self.num_goals, False)
        
        # For now, hard coding termination for when you arrive at the goal state
        # when following the option for that goal
        for g in range(self.num_goals):
            action, term = self._get_goal_policy_term(xp, g)
            x_action, _ = self._get_goal_policy_term(x, g)
            if action is not None:
                option_pi_xp[g, action] = 1
                
            if x_action is not None:
                valid_goal[g] = True

            x_option_gamma[g] = (1 - term)

        self.state_estimate_learner.update(x, a, xp, r, gamma, option_pi_xp, x_option_gamma, self.step_size, valid_goal)
        pass

    def _map_goal_to_state_estimates(self, x, a, xp, r, gamma):
        option_pi_x = np.zeros((self.num_goals, self.num_actions))
        for g in range(self.num_goals):
            action, term = self._get_goal_policy_term(x, g)
            if action is not None:
                option_pi_x[g, action] = 1
    
        self.goal_estimate_learner.update(x, option_pi_x, self.state_estimate_learner.r[x], self.state_estimate_learner.gamma[x], self.step_size, r, xp, self.goal_func)

    def _improve_goal_values(self):
        goal_reward = self.goal_estimate_learner.r
        goal_reward = (goal_reward + self.kappa * np.sqrt(self.tau))
        # print(goal_reward)
        # goal_reward = goal_reward.T
        # print(goal_reward)
        self.goal_value_learner.update(self.goal_estimate_learner.gamma, goal_reward, self.goal_estimate_learner.goal_r, globals.param['gamma'])
    
    def _option_constrained_improvement(self, x, xp,):
        goal_values = self.goal_value_learner.goal_values
        goal_reward = self.state_estimate_learner.r
        goal_enter_reward = self.goal_estimate_learner.goal_r
        goal_enter_reward = (goal_enter_reward + self.kappa * np.sqrt(self.tau))
        goal_gamma = self.state_estimate_learner.gamma

        # Plan with only the current state for now. Only need 1 sample
        for _x in self.search_control.sample_states(1, self.history_dict, x, xp):

                # returns = reward_goals[g] + goal_gamma[g] / gamma * goal_r + goal_gamma[g] * self.goal_values
                # valid_goals = np.nonzero(goal_gamma[g])

                # if len(valid_goals[0]) > 0:
                #     self.goal_values[g] = np.max(returns[valid_goals[0]])
                    

            # Small hack here to get a factor of gamma off the gamma
            option_values = goal_reward[_x] + goal_gamma[_x] /  globals.param['gamma']  * goal_enter_reward  + goal_gamma[_x] * goal_values

            # print(option_values.shape)
            # print(goal_gamma[_x].shape)
            invalid_indices = np.where(goal_gamma[_x] == 0)
            option_values[invalid_indices] = -np.inf

            best_option_values = np.max(option_values, axis=1)

            # print(best_option_values)

            for a in range(self.num_actions):
                if best_option_values[a] != -np.inf:
                    self.behaviour_learner.Q[_x, a] += self.step_size * (best_option_values[a] - self.behaviour_learner.Q[_x, a])
                    # print(_x)

            # for a in range(self.num_actions):

            #     # If there are valid goals
            #     if len(np.nonzero(goal_gamma[_x][a])[0]) > 0: 
                    
  
            #         # print(option_values)
            #         best_option_values = np.max(option_values)
            #         # print(best_option_values)
                    
                    # self.behaviour_learner.Q[_x, a] += self.step_size * (best_option_values - self.behaviour_learner.Q[_x, a])

    def agent_end(self, s, a, r, gamma):
        self.update(s, a, None, r, gamma, terminal=True)
        self.behaviour_learner.episode_end()

class GoalPolicyLearner():
    def __init__(self):
        pass
class ESARSAStateEstimates:
    def __init__(self, num_states, num_actions, num_goals):
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_goals = num_goals

        # initializing weights
        self.r = np.zeros((self.num_states, self.num_actions, self.num_goals))
        self.gamma = np.zeros((self.num_states, self.num_actions, self.num_goals))
    
    def update(self, x, a, xp, r, gamma, option_pi_xp, option_gamma, alpha, valid_state_for_goal):
        for g in range(self.num_goals):
            # Check for in the start state distribution
            if valid_state_for_goal[g]:
                
                # Getting full
                # self.r[x, a, g] += alpha * (r + gamma * option_gamma[g] * np.sum(option_pi_xp[g] * self.r[xp, :, g]) - self.r[x, a, g])
                if option_gamma[g] != 0:
                    self.r[x, a, g] += alpha * (r + gamma * option_gamma[g] * np.sum(option_pi_xp[g] * self.r[xp, :, g]) - self.r[x, a, g])
                else:
                    self.r[x, a, g] += alpha * (0 - self.r[x, a, g])
                self.gamma[x, a, g] += alpha * (gamma * (1 - option_gamma[g]) + gamma * option_gamma[g] * np.sum(option_pi_xp[g] * self.gamma[xp, :, g]) - self.gamma[x, a, g])

        def log():
            globals.collector.collect('state_r', np.copy(self.r))
            globals.collector.collect('state_gamma', np.copy(self.gamma))
        run_if_should_log(log)
            
class GoalEstimates:
    def __init__(self, num_goals):
        self.num_goals = num_goals

        # initializing weights
        self.r = np.zeros((self.num_goals, self.num_goals))
        self.gamma = np.zeros((self.num_goals, self.num_goals))
        self.goal_r = np.zeros(self.num_goals)

    def update(self, x, option_pi_x, r_s, gamma_s, alpha, r, xp,  goal_func):
        x_on_goals = goal_func(x)
        xp_on_goal = goal_func(xp)

        for g in range(self.num_goals):
            if x_on_goals[g] == 1: 
                self.r[g] += alpha * (np.sum(option_pi_x * r_s.T, axis=1)- self.r[g])
                self.gamma[g] += alpha * (np.sum(option_pi_x * gamma_s.T, axis=1)- self.gamma[g])
            
            if xp_on_goal[g] == 1:
                self.goal_r[g] += alpha * (r - self.goal_r[g])

        def log():
            globals.collector.collect('goal_r', np.copy(self.r))
            globals.collector.collect('goal_gamma', np.copy(self.gamma))
        run_if_should_log(log)

class GoalValueLearner:
    def __init__(self, num_goals):
        self.num_goals = num_goals
        
        # Initializing goal values
        self.goal_values = np.zeros(self.num_goals)
    
    def update(self, goal_gamma, reward_goals, goal_r, gamma):
        num_planning_steps = 2
        for _ in range(num_planning_steps):
            # Just doing value iteration for now 

            # Can we possibly vectorize this?
            for g in range(self.num_goals):
                # Dividing by gamma as a hacky way of getting gamma for step - 1
                returns = reward_goals[g] + goal_gamma[g] / gamma * goal_r + goal_gamma[g] * self.goal_values
                valid_goals = np.nonzero(goal_gamma[g])

                if len(valid_goals[0]) > 0:
                    self.goal_values[g] = np.max(returns[valid_goals[0]])
                    
            # Old vectorized equation. Doesn't account for whether goal values are valid.
            # self.goal_values = np.max(reward_goals + goal_gamma * self.goal_values, axis=1)
        # print(self.goal_values)
        
        if globals.blackboard['num_steps_passed'] % globals.blackboard['step_logging_interval'] == 0:
            globals.collector.collect('goal_values', np.copy(self.goal_values)) 