from typing import Dict, Union, Tuple
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
from src.utils import numpy_utils
from src.utils.log_utils import run_if_should_log

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    # Important for forward reference
    from src.problems.BaseProblem import BaseProblem

# Filter classes
class ConstantFilter():
    def __init__(self, prob: float, rng):
        self.rng = rng
        self.prob = prob

        assert prob >= 0.0 and prob <= 1.0, f'Probability must be between 0.0 and 1.0. Instead, prob is {self.prob}'

    def update(self, x: int, x_action_value):
        pass

    def should_filter(self, x: int):
        rand = self.rng.random()
        if rand < self.prob:
            return True
        return False

class CommonGreedyFilter():
    def __init__(self, rng, greedy_policy_lr: float, filter_weight_lr: float, num_states: int, num_actions: int, init_weight: float = 0):
        self.greedy_policy_lr = greedy_policy_lr
        self.filter_weight_lr = filter_weight_lr
        self.num_states = num_states
        self.num_actions = num_actions

        # setting the starting avg greedy policy to be random
        self.avg_greedy_policy = np.ones((num_states, num_actions)) / num_actions

        self.filter_weights = np.full(num_states, init_weight, dtype=np.float32)

        self.rng = rng

    def update(self, x: int, x_action_value):
        greedy_action = np.argmax(x_action_value)
        self.avg_greedy_policy[x] += self.greedy_policy_lr * (numpy_utils.create_onehot(self.num_actions, greedy_action) - self.avg_greedy_policy[x])

        avg_greedy_policy = self.avg_greedy_policy[x]
        # print(avg_greedy_policy)
        distance = np.sum(avg_greedy_policy * -1) + 2 * avg_greedy_policy[greedy_action] 

        # print(distance)
        self.filter_weights[x] += self.filter_weight_lr * distance

        def log():
            globals.collector.collect('skip_probability_weights', np.copy(self.filter_weights))
            globals.collector.collect('avg_greedy_policy', np.copy(self.avg_greedy_policy))

        run_if_should_log(log)

    def should_filter(self, x):
        rand = self.rng.random()
        sig = 1/(1 + np.exp(np.clip(-self.filter_weights[x], -1e2, 1e2)))

        if rand < sig:
            return True
        return False

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

        # Temporal filter hyperparams
        self.base_q = param_utils.parse_param(params, 'base_q', lambda p : isinstance(p, bool), optional=True, default=False)

        self.use_optimal_skip = param_utils.parse_param(params, 'use_optimal_skip', lambda p : isinstance(p, bool), optional=True, default=False)
        self.use_optimal_options = param_utils.parse_param(params, 'use_optimal_options', lambda p : isinstance(p, bool), optional=True, default=False)


        self.filter_class = param_utils.parse_param(params, 'filter_class', lambda p: p in ['constant', 'common_greedy'])

        if self.filter_class == 'constant':
            self.filter_prob = param_utils.parse_param(params, 'filter_prob', lambda p: p >= 0)
            self.filter = ConstantFilter(self.filter_prob, self.random)
        elif self.filter_class == 'common_greedy':
            self.skip_lr = param_utils.parse_param(params, 'skip_lr', lambda p : isinstance(p, float), optional=True, default=1e-3)
            self.avg_greedy_policy_lr = param_utils.parse_param(params, 'avg_greedy_policy_lr', lambda p : isinstance(p, float), optional=True, default=1e-3)
            self.initial_skip_weight = param_utils.parse_param(params, 'initial_skip_weight', optional=True, default=-4)

            self.filter = CommonGreedyFilter(self.random, self.avg_greedy_policy_lr, self.skip_lr, self.num_states, self.num_actions, self.initial_skip_weight)
            

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
        if self.use_optimal_skip:
            if x != 7:
                return True
            else:
                return False

        return self.filter.should_filter(x)
    
    def skip_update(self, x, a, xp, r, gamma, terminal):

        # self.behaviour_learner.update(x, a, xp, r, gamma, self.alpha)

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

        self.filter.update(x, self.behaviour_learner.get_action_values(x))

        if not terminal:
            ap = self.selectAction(sp)
        else:
            ap = None

        self.cumulative_reward += r
        if globals.blackboard['num_steps_passed'] % globals.blackboard['step_logging_interval'] == 0:
            globals.collector.collect('Q', np.copy(self.behaviour_learner.Q)) 
            
            globals.collector.collect('reward_rate', np.copy(self.cumulative_reward) / globals.blackboard['step_logging_interval'])
            self.cumulative_reward = 0

        return ap
    def agent_end(self, s, a, r, gamma):
        self.update(s, a, None, r, gamma, terminal=True)
        self.behaviour_learner.episode_end()

        
        # Resetting skip vars
        self.skip_x = None
        self.skip_a = None
        self.skip_r = -1
        self.skip_gamma = -1