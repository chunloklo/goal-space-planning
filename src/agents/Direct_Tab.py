from typing import Dict, Union, Tuple
import numpy as np
from numpy.lib.function_base import average
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
from src.utils import numpy_utils

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    # Important for forward reference
    from src.problems.BaseProblem import BaseProblem

class Direct_Tab:
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

        # This is only needed for RL glue (for convenience reason to have uniform initialization between
        # options and non-options agents). Not used in the algorithm at all.
        self.options = options
        params = self.params
        # define parameter contract
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']
        self.gamma = params['gamma']
        self.behaviour_alg = param_utils.parse_param(params, 'behaviour_alg', lambda p : p in ['QLearner', 'ESarsaLambda']) 

        self.skip_action = param_utils.parse_param(params, 'skip_action', lambda p : isinstance(p, bool))
        self.base_q = param_utils.parse_param(params, 'base_q', lambda p : isinstance(p, bool), optional=True, default=False)

        self.use_optimal_skip = param_utils.parse_param(params, 'use_optimal_skip', lambda p : isinstance(p, bool), optional=True, default=False)
        self.use_optimal_options = param_utils.parse_param(params, 'use_optimal_options', lambda p : isinstance(p, bool), optional=True, default=False)

        # +1 accounts for the terminal state
        if self.behaviour_alg == 'QLearner':
            self.behaviour_learner = QLearner(self.num_states + 1, self.num_actions)
        else:
            raise NotImplementedError(f'behaviour_alg for {self.behaviour_alg} is not implemented')

        # Initial variables for logging
        self.skip_x = None
        self.skip_a = None
        self.skip_r = -1
        self.skip_gamma = -1

        # Logging
        self.cumulative_reward = 0

        self.skip_lr = param_utils.parse_param(params, 'skip_lr', lambda p : isinstance(p, float), optional=True, default=1e-3)

        self.avg_greedy_policy = np.ones((self.num_states, self.num_actions)) / self.num_actions

        self.initial_skip_weight = param_utils.parse_param(params, 'initial_skip_weight', optional=True, default=-8)
        self.skip_probability_weights = np.ones((self.num_states)) * self.initial_skip_weight

    def FA(self):
        return "Tabular"

    def __str__(self):
        return "Dyna_Tab"

    def get_policy(self, x: int) -> npt.ArrayLike:
        probs = np.zeros(self.num_actions)
        probs += self.epsilon / (self.num_actions)

        action_values = self.behaviour_learner.get_action_values(x)
        a = np.argmax(action_values)

        probs[a] += 1 - self.epsilon
        return probs

    # public method for rlglue
    def selectAction(self, s: Any) -> int:
        x = self.representation.encode(s)

        if self.use_optimal_options:
            # UP = 0
            # RIGHT = 1
            # DOWN = 2
            # LEFT = 3

            if s[1] == 0 or s[1] == self.env.size - 1:
                return 2

            if s[1] == (self.env.size - 1) // 2 and s[0] != 0:
                return 0

            return 0

        # if not self.skip_action:
        return self.random.choice(self.num_actions, p = self.get_policy(x))

    def should_skip(self, x) -> bool:

        if not self.skip_action:
            return False


        if self.use_optimal_skip:
            if x != 7:
                return True
            else:
                return False

        sig = 1/(1 + np.exp(-self.skip_probability_weights[x]))

        # min_prob = 0.1 
        # sig = np.clip(sig, a_min = 0, a_max = 1 - min_prob)


        rand = self.random.random()
        # print(sig, rand)
        if rand < sig:
            # print('skipping')
            return True

        return False
    
    def skip_update(self, x, a, xp, r, gamma, terminal):

        self.behaviour_learner.update(x, a, xp, r, gamma, self.alpha)

        if self.base_q:
            return
    
        # Second layer temporally extended update
        skip = self.should_skip(x)

        # print(f'x: {x}')

        if not skip:
            if self.skip_x is not None:
                # First update
                # print(f'skip update {self.skip_x} {self.skip_a} {x}')
                self.behaviour_learner.update(self.skip_x, self.skip_a, x, self.skip_r, self.skip_gamma, self.alpha)

            # Caching x and a for later update
            self.skip_x = x
            self.skip_a = a
            self.skip_r = r
            self.skip_gamma = gamma
        else:
            # print(f'skipped x: {x}')
            self.skip_r += self.skip_gamma * r
            self.skip_gamma *= gamma

        if terminal:
            if skip and self.skip_x is not None:
                # print(f'terminal skipped {self.skip_x} {self.skip_a} {xp} r {self.skip_r} gamma {self.skip_gamma}')
                self.behaviour_learner.update(self.skip_x, self.skip_a, xp, self.skip_r, self.skip_gamma, self.alpha)



    def update(self, s, a, sp, r, gamma, terminal: bool = False) -> int:
        x = self.representation.encode(s)
        # Treating the terminal state as an additional state in the tabular setting
        xp = self.representation.encode(sp) if not terminal else self.num_states

        self.old_Q = np.copy(self.behaviour_learner.Q)

        self.skip_update(x, a, xp, r, gamma, terminal)
        # self.behaviour_learner.update(x, a, xp, r, gamma, self.alpha)

        greedy_action =  np.argmax(self.behaviour_learner.Q[x])

        self.avg_greedy_policy[x] += self.skip_lr * (numpy_utils.create_onehot(self.num_actions, greedy_action) - self.avg_greedy_policy[x])

        # avg_greedy_policy = self.avg_greedy_policy[x] / np.sum(self.avg_greedy_policy[x])
        avg_greedy_policy = self.avg_greedy_policy[x]
    
        distance = np.sum(avg_greedy_policy * -1) + 2 * avg_greedy_policy[greedy_action]    
        # distance = 1 if np.argmax(avg_greedy_policy) == greedy_action else -1

        self.skip_probability_weights[x] += self.skip_lr * distance

        # if (x == 0):
        #     print(f'avg greedy : {avg_greedy_policy}, greedy: {greedy_action} distance: {distance} skip_weight: {self.skip_probability[x]}')
        
        if not terminal:
            ap = self.selectAction(sp)
        else:
            ap = None

        self.cumulative_reward += r
        if globals.blackboard['num_steps_passed'] % globals.blackboard['step_logging_interval'] == 0:
            globals.collector.collect('Q', np.copy(self.behaviour_learner.Q)) 
            
            globals.collector.collect('reward_rate', np.copy(self.cumulative_reward) / globals.blackboard['step_logging_interval'])
            self.cumulative_reward = 0

            sig = 1/(1 + np.exp(-self.skip_probability_weights))
            globals.collector.collect('skip_probability_weights', np.copy(self.skip_probability_weights))

            globals.collector.collect('avg_greedy_policy', np.copy(self.avg_greedy_policy))

        return ap
    def agent_end(self, s, a, r, gamma):
        self.update(s, a, None, r, gamma, terminal=True)
        self.behaviour_learner.episode_end()

        
        # Resetting skip vars
        self.skip_x = None
        self.skip_a = None
        self.skip_r = -1
        self.skip_gamma = -1