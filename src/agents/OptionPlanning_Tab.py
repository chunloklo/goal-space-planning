from typing import Dict, Optional, Union, Tuple
import numpy as np
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

from src.utils.numpy_utils import create_onehot
# from src.environments.GrazingWorldAdam import get_pretrained_option_model, state_index_to_coord, get_all_transitions

if TYPE_CHECKING:
    # Important for forward reference
    from src.problems.BaseProblem import BaseProblem

class OptionPlanning_Tab:
    def __init__(self, problem: 'BaseProblem'):
        self.wrapper_class = rlglue.OneStepWrapper

        self.env = problem.getEnvironment()
        self.representation: Tabular = problem.get_representation('Tabular')
        self.features = self.representation.features
        self.num_actions = problem.actions
        self.actions = list(range(self.num_actions))
        self.params = problem.params
        self.num_states = self.env.nS
        self.random = np.random.RandomState(problem.seed)

        self.options = problem.options
        self.num_options = len(problem.options)

        # This is only needed for RL glue (for convenience reason to have uniform initialization between
        # options and non-options agents). Not used in the algorithm at all.
        params = self.params

        # define parameter contract
        self.alpha = params['alpha'] # general step size
        self.epsilon = params['epsilon'] # exploration parameter
        self.planning_steps = params['planning_steps']
        self.goal_planning_steps = params['goal_planning_steps']
        self.gamma = params['gamma']
        self.q_init = param_utils.parse_param(params, 'q_init', lambda p: True, default=0, optional=True)

        
        self.search_control = ActionModelSearchControl_Tabular('current', self.random)
        # This action model is only for random search control
        self.action_model = DictModel()
         
        self.a = -1
        self.x = -1

        # +1 accounts for the terminal state
        self.behaviour_learner = QLearner(self.num_states + 1, self.num_actions)
        self.behaviour_learner.Q[:] = self.q_init

        # Supposed all goals have the same termination set for now
        self.goals = self.options[0].termination_set
        print(self.goals)

        # State estimate learner
        self.state_estimate_learner = ESARSAStateEstimates(self.num_states + 1, self.num_actions, self.num_options, len(self.goals))

        def state_in_goal_func(x):
            return np.array([x == goal for goal in self.goals])

        # Goal estimate learner
        self.goal_estimate_learner = GoalEstimates(self.num_options, len(self.goals), state_in_goal_func)

        self.goal_value_learner = GoalValueLearner(len(self.goals))

        # For logging state visitation
        self.cumulative_reward = 0

    def FA(self):
        return "Tabular"

    def __str__(self):
        return "OptionPlanning_Tab"

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

        # Direct RL Update
        self.behaviour_learner.update(x, a, xp, r, gamma, self.alpha)

        # if (r > 0):
            # print(x)
            # print(self.behaviour_learner.get_action_values(x))

        # Updating search control
        self.action_model.update(x, a, xp, r, gamma)
        self.search_control.update(x, xp)
        
        self._update_state_estimates(x, a, xp, r, gamma)
        self._map_goal_to_state_estimates(x, a, xp, r, gamma)
        self._improve_goal_values()
        self._option_constrained_improvement(x, xp)

        if not terminal:
            ap = self.selectAction(sp)
        else:
            ap = None

        self.cumulative_reward += r
        if globals.blackboard['num_steps_passed'] % globals.blackboard['step_logging_interval'] == 0:
            globals.collector.collect('Q', np.copy(self.behaviour_learner.Q)) 
            globals.collector.collect('reward_rate', np.copy(self.cumulative_reward) / globals.blackboard['step_logging_interval'])

            self.cumulative_reward = 0

        # print(sp, ap)
        return ap

    def get_policy(self, x: int) -> npt.ArrayLike:
        # Implements epsilon greedy here
        probs = np.zeros(self.num_actions)
        probs += self.epsilon / (self.num_actions)
        a = np.argmax(self.behaviour_learner.get_action_values(x))
        probs[a] += 1 - self.epsilon
        return probs

    def _update_state_estimates(self, x, a, xp, r, gamma):
        option_pi_xp = np.zeros((self.num_options, self.num_actions))
        option_gamma = np.zeros(self.num_options)
        option_goal_terminations = np.zeros((self.num_options, len(self.goals)))

        goal_termination = xp == np.array(self.goals)
        for o, option in enumerate(self.options):
            if xp is self.num_states:
                # Have the option always take a specific action if xp terminates
                action, term = 0, True
            else:
                action, term = option.step(xp)

            option_pi_xp[o, action] = 1
            option_gamma[o] = (1 - term)

            if term:
                option_goal_terminations[o] = goal_termination

        self.state_estimate_learner.update(x, a, xp, r, gamma, option_pi_xp, option_gamma, self.alpha, option_goal_terminations)
        pass

    def _map_goal_to_state_estimates(self, x, a, xp, r, gamma):
        option_pi_x = np.zeros((self.num_options, self.num_actions))
        for o, option in enumerate(self.options):
            if xp is self.num_states:
                # Have the option always take a specific action if xp terminates
                action, term = 0, True
            else:
                action, term = option.step(x)

            option_pi_x[o, action] = 1

        self.goal_estimate_learner.update(x, a, r, xp, gamma, option_pi_x, 
            self.state_estimate_learner.r[x], self.state_estimate_learner.gamma[x], self.state_estimate_learner.goal_transition_prob[x], 
            self.behaviour_learner.get_action_values(x), self.alpha)

    def _improve_goal_values(self):
        self.goal_value_learner.update(self.goal_estimate_learner.value_baseline, self.goal_estimate_learner.goal_transition_prob, 
            self.goal_estimate_learner.gamma, self.goal_estimate_learner.r, self.alpha)
    
    def _option_constrained_improvement(self, x, xp):
        goal_values = self.goal_value_learner.goal_values
        goal_transition_prob = self.state_estimate_learner.goal_transition_prob
        goal_reward = self.state_estimate_learner.r
        goal_gamma = self.state_estimate_learner.gamma

        # print(goal_values.shape)
        # print(goal_transition_prob.shape)
        # print(goal_reward.shape)
        # print(goal_gamma.shape)
        # asdas

        # Plan with only the current state for now. Only need 1 sample
        for _x in self.search_control.sample_states(1, self.action_model, x, xp):
            # print((goal_transition_prob[_x] * goal_values).shape)
            # asa
            option_values = goal_reward[_x] + goal_gamma[_x] * np.sum(goal_transition_prob[_x] * goal_values, axis=2)
            # print(option_values.shape)
            
            best_option_values = np.max(option_values, axis=1)

            self.behaviour_learner.Q[_x] += self.alpha * (best_option_values - self.behaviour_learner.Q[_x])

            # asd
        # asdas

    def agent_end(self, s, a, r, gamma):
        self.update(s, a, None, r, gamma, terminal=True)
        self.behaviour_learner.episode_end()

class ESARSAStateEstimates:
    def __init__(self, num_states, num_actions, num_options, num_goals):
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_options = num_options
        self.num_goals = num_goals

        # initializing weights
        self.r = np.zeros((self.num_states, self.num_actions, self.num_options))
        self.gamma = np.zeros((self.num_states, self.num_actions, self.num_options))
        self.goal_transition_prob = np.zeros((self.num_states, self.num_actions, self.num_options, self.num_goals))
    
    def update(self, x, a, xp, r, gamma, option_pi_xp, option_gamma, alpha, goal_termination):
        for o in range(self.num_options):
            # print('start debug')
            # print(f'r {r} gamma {gamma} option_gamma {option_gamma[o]} option_pi_xp {option_pi_xp[o]} r_xp {self.r[xp, :, o]}')
            # print(np.sum(option_pi_xp[o] * self.r[xp, :, o]))
            # print(r + gamma * option_gamma[o] + np.sum(option_pi_xp[o] * self.r[xp, :, o]))
            # print(xp)
            # sdfsdf
            self.r[x, a, o] += alpha * (r + gamma * option_gamma[o] * np.sum(option_pi_xp[o] * self.r[xp, :, o]) - self.r[x, a, o])

            self.gamma[x, a, o] += alpha * ((1 - option_gamma[o]) + gamma * option_gamma[o] * np.sum(option_pi_xp[o] * self.gamma[xp, :, o]) - self.gamma[x, a, o])
            
            option_goal_termination = goal_termination[o]

            # some tricky dimension manipulation to get broadcasting correct 
            goal_transition_target = option_goal_termination + option_gamma[o] * np.sum((self.goal_transition_prob[xp, :, o].T * option_pi_xp[o]), axis=1)
            self.goal_transition_prob[x, a, o] += alpha * (goal_transition_target - self.goal_transition_prob[x, a, o])

        if globals.blackboard['num_steps_passed'] % globals.blackboard['step_logging_interval'] == 0:
            globals.collector.collect('state_estimate_r', np.copy(self.r)) 
            globals.collector.collect('state_estimate_gamma', np.copy(self.gamma)) 
            globals.collector.collect('state_estimate_goal_prob', np.copy(self.goal_transition_prob)) 

class GoalEstimates:
    def __init__(self, num_options, num_goals, goal_func):
        self.num_options = num_options
        self.num_goals = num_goals
        self.goal_func = goal_func

        # initializing weights
        self.r = np.zeros((self.num_goals, self.num_options))
        self.gamma = np.zeros((self.num_goals, self.num_options))
        self.goal_transition_prob = np.zeros((self.num_goals, self.num_options, self.num_goals))
        self.value_baseline = np.zeros(self.num_goals)

    def update(self, x, a, r, xp, gamma, option_pi_x, r_s, gamma_s, goal_prob_s, q_sa, alpha):
        on_goals = self.goal_func(x)
        for g in range(self.num_goals):
            if on_goals[g] == 1:    
                self.r[g] += alpha * (np.sum(option_pi_x * r_s.T, axis=1)- self.r[g])
                self.gamma[g] += alpha * (np.sum(option_pi_x * gamma_s.T, axis=1)- self.gamma[g])
                self.goal_transition_prob[g] += alpha * (np.sum((goal_prob_s.T * option_pi_x).T, axis=0) - self.goal_transition_prob[g])
                self.value_baseline[g] += alpha * (np.max(q_sa) - self.value_baseline[g])


class GoalValueLearner:
    def __init__(self, num_goals):
        self.num_goals = num_goals
        
        # Initializing goal values
        self.goal_values = np.zeros(self.num_goals)
    
    def update(self, value_baseline, goal_transition_prob, goal_gamma, reward_goals, alpha):
        
        num_planning_steps = 2

        for _ in range(num_planning_steps):
        #     # Literally doing value iteration here
            goal_transition_values = reward_goals + goal_gamma * np.sum(goal_transition_prob * self.goal_values, axis=2)
            self.goal_values = np.maximum(np.max(goal_transition_values, axis=1), value_baseline)
        pass

        if globals.blackboard['num_steps_passed'] % globals.blackboard['step_logging_interval'] == 0:
            globals.collector.collect('goal_values', np.copy(self.goal_values)) 