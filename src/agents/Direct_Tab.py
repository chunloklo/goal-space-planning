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

        self.one_step_suboptimality = np.zeros(self.num_states)
        self.suboptimality = np.zeros(self.num_states)
        self.best_action = np.zeros((self.num_states, self.num_actions))

        self.avg_suboptimality = np.zeros(self.num_states)
        self.num_avg_suboptimality_updated = np.zeros(self.num_states)

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
        a = self.random.choice(self.num_actions, p = self.get_policy(x))
        return a

    
    def skip_update(self, x, a, xp, r, gamma, terminal):

        # print(f'{x} {a} {xp}')

        # Skip logic for state x
        skip = False
        # if x == 38:
            # skip = True

        # if x == 42:
            # print(self.avg_suboptimality[x])
        # if x == 3:
            # print(self.avg_suboptimality[x])
        if self.avg_suboptimality[x] < 0.2:
            skip = True

        # print(self.avg_suboptimality[x])
        

        if self.skip_x is None:
            # This is the first update.
            skip = False

        if not skip:
            if self.skip_x is not None:
                # First update
                print(f'not skipped {self.skip_x} {self.skip_a} {xp}')
                self.behaviour_learner.update(self.skip_x, self.skip_a, x, self.skip_r, self.skip_gamma, self.alpha)

            # Caching x and a for later update
            self.skip_x = x
            self.skip_a = a
            self.skip_r = r
            self.skip_gamma = gamma
        else:
            print(f'skipped {x} {a} {xp}')
            self.behaviour_learner.update(x, a, xp, r, gamma, self.alpha)
            self.skip_r += self.skip_gamma * r
            self.skip_gamma *= gamma
        
        if terminal:
            if skip:
                print(f'terminal skipped {self.skip_x} {self.skip_a} {xp}')
                self.behaviour_learner.update(self.skip_x, self.skip_a, xp, self.skip_r, self.skip_gamma, self.alpha)
            else:
                print(f'terminal not skipped {x} {a} {xp}')
                self.behaviour_learner.update(x, a, xp, r, gamma, self.alpha) 

    def update(self, s, a, sp, r, gamma, terminal: bool = False) -> int:
        x = self.representation.encode(s)
        # Treating the terminal state as an additional state in the tabular setting
        xp = self.representation.encode(sp) if not terminal else self.num_states
        
        self.old_Q = np.copy(self.behaviour_learner.Q)

        self.skip_update(x, a, xp, r, gamma, terminal)

        if not terminal:
            ap = self.selectAction(sp)
        else:
            ap = None

        # Logging one-step suboptimality
        best_prev_action = np.argmax(self.old_Q[x])
        best_new_action = np.argmax(self.behaviour_learner.Q[x])

        self.best_action[x, best_prev_action] += 1

        # print(self.behaviour_learner.Q[x][best_new_action] - self.behaviour_learner.Q[x][best_prev_action])
        avg_prev_action = self.best_action[x] / np.sum(self.best_action[x])

        self.one_step_suboptimality[x] += self.behaviour_learner.Q[x][best_new_action] - self.behaviour_learner.Q[x][best_prev_action]

        if self.behaviour_learner.Q[x][best_new_action] > 0:
            # Baselining suboptimality at 0
            self.suboptimality[x] += (self.behaviour_learner.Q[x][best_new_action] - np.sum(self.behaviour_learner.Q[x] * avg_prev_action)) / self.behaviour_learner.Q[x][best_new_action]

            suboptimality = (self.behaviour_learner.Q[x][best_new_action] - np.sum(self.behaviour_learner.Q[x] * avg_prev_action)) / self.behaviour_learner.Q[x][best_new_action]
            # print(suboptimality)
            # avg suboptimality
            self.num_avg_suboptimality_updated[x] += 1
            self.avg_suboptimality[x] += (suboptimality - self.avg_suboptimality[x]) / self.num_avg_suboptimality_updated[x]

            # print(self.avg_suboptimality[x])


        self.cumulative_reward += r
        if globals.blackboard['num_steps_passed'] % globals.blackboard['step_logging_interval'] == 0:
            globals.collector.collect('Q', np.copy(self.behaviour_learner.Q)) 
            
            globals.collector.collect('reward_rate', np.copy(self.cumulative_reward) / globals.blackboard['step_logging_interval'])
            self.cumulative_reward = 0

            globals.collector.collect('one_step_suboptimality', np.copy(self.one_step_suboptimality) / globals.blackboard['step_logging_interval'])

            globals.collector.collect('suboptimality', np.copy(self.suboptimality) / globals.blackboard['step_logging_interval'])
            self.suboptimality[:] = 0

            globals.collector.collect('avg_suboptimality', np.copy(self.avg_suboptimality))

        return ap
    def agent_end(self, s, a, r, gamma):
        self.update(s, a, None, r, gamma, terminal=True)
        self.behaviour_learner.episode_end()

        
        # Resetting skip vars
        self.skip_x = None
        self.skip_a = None
        self.skip_r = -1
        self.skip_gamma = -1