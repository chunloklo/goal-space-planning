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
        self.slow_alpha = params['slow_alpha']
        self.epsilon = params['epsilon']
        self.gamma = params['gamma']
        self.lmbda = params['lambda']
        self.skip_prob = params['skip_prob']
        self.behaviour_alg = param_utils.parse_param(params, 'behaviour_alg', lambda p : p in ['QLearner', 'ESarsaLambda']) 
        self.skip_alg = param_utils.parse_param(params, 'skip_alg', lambda p : p in ['Testing', 'None', 'Learning'], default='None', optional=False) 
        # self.test_alg = param_utils.parse_param(params, 'test_alg', lambda p : p in ['skip_state', None], default=None, optional=True) 
        self.a = -1
        self.x = -1

        # +1 accounts for the terminal state
        if self.behaviour_alg == 'QLearner':
            self.behaviour_learner = QLearner(self.num_states + 1, self.num_actions)
            self.slow_behaviour_learner = QLearner(self.num_states + 1, self.num_actions)
        elif self.behaviour_alg == 'ESarsaLambda':
            self.behaviour_learner = ESarsaLambda(self.num_states + 1, self.num_actions)
        else:
            raise NotImplementedError(f'behaviour_alg for {self.behaviour_alg} is not implemented')

        # For logging state visitation
        self.state_visitations = np.zeros(self.num_states)

        self.policy_agreement = np.zeros(self.num_states + 1)
        self.policy_agreement_step_size = 0.0005

        self.average_policy = np.zeros((self.num_states, self.num_actions))
        self.learning_rate = 0.0005
        
        self.policy_switch = np.zeros((self.num_states + 1, self.num_actions))
        self.policy_switch += 4
        self.average_learning = np.zeros(self.num_states + 1)
        self.average_policy_distance = np.zeros(self.num_states + 1)

        self.interim_r = 0
        self.prior_x = None
        self.prior_a = None

        self.skipped_states = np.zeros(self.num_states)

        self.mean_learning = 0

        self.cumulative_reward = 0

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

    def learning_update(self, x, a, xp, r, gamma):
        if isinstance(self.behaviour_learner, QLearner):
            self.behaviour_learner.update(x, a, xp, r, gamma, self.alpha)
            self.slow_behaviour_learner.update(x, a, xp, r, gamma, self.slow_alpha)
        elif isinstance(self.behaviour_learner, ESarsaLambda):
            self.behaviour_learner.update(x, a, xp, r, self.get_policy(xp), self.gamma, self.lmbda, self.alpha)
        else:
            raise NotImplementedError()

        behaviour_a = np.argmax(self.behaviour_learner.get_action_values(x))
        slow_behaviour_a = np.argmax(self.slow_behaviour_learner.get_action_values(x))

        agreement = 1 if behaviour_a == slow_behaviour_a else 0
        self.policy_agreement[x] += self.policy_agreement_step_size * (agreement - self.policy_agreement[x])

    def update(self, s, a, sp, r, gamma, terminal: bool = False) -> int:
        x = self.representation.encode(s)
        # Treating the terminal state as an additional state in the tabular setting
        xp = self.representation.encode(sp) if not terminal else self.num_states
        # print(f'x:{x} a: {a} xp:{xp}')

        # Initialization
        if self.prior_x == None:
            self.prior_x = x
            self.prior_a = a

        self.state_visitations[x] += 1

        self.mean_learning = (1 - self.learning_rate) * self.mean_learning + self.learning_rate * self.average_learning[x] 
        prior_q_values = np.copy(self.behaviour_learner.get_action_values(x))
        prior_best_a = np.argmax(prior_q_values)

        # self.learning_update(x, a, xp, r, gamma)

        skip_condition = False
        if self.skip_alg == 'Testing':
            # if self.random.uniform(0, 1) < self.policy_agreement[x]:
            #     skip_condition = True
            if xp not in [self.env.size // 2]:
                if self.random.uniform(0, 1) < self.skip_prob:
                    skip_condition = True
        
        if self.skip_alg == 'Learning':
            pass

        if terminal:
            skip_condition = False

        if skip_condition:
            self.skipped_states[xp] += 1
            self.interim_r += r
        else:
            # swapping prior_x and x
            # print(f'x:{x} prior_x:{self.prior_x} xp:{xp}')
            learn_x = self.prior_x
            learn_a = self.prior_a
            learn_r = r + self.interim_r
            self.interim_r = 0
            self.learning_update(learn_x, learn_a, xp, learn_r, gamma)

            # if learn_x == 32:
            #     print(f'x:{learn_x} a:{learn_a} xp:{xp} gamma:{gamma}')

        new_q_values = self.behaviour_learner.get_action_values(x)

        best_a = np.argmax(self.behaviour_learner.get_action_values(x))
        self.average_policy[x, :] = (1 - self.learning_rate) * self.average_policy[x, :] + self.learning_rate * numpy_utils.create_onehot(self.num_actions, best_a)

        # sdfsdf
        average_policy  = self.average_policy[x, :] / np.linalg.norm(self.average_policy[x, :], ord=1)
        current_policy = numpy_utils.create_onehot(self.num_actions, best_a)
        policy_distance = np.sum(np.abs(average_policy - current_policy))
        self.average_policy_distance[x] = (1 - self.learning_rate) * self.average_policy_distance[x] + policy_distance * self.learning_rate

        prior_policy_prob = numpy_utils.create_onehot(self.num_actions, prior_best_a)

        average_value = np.sum(average_policy * self.behaviour_learner.get_action_values(x))
        # print(value)
        best_value = self.behaviour_learner.get_action_values(x)[best_a]
        # print(best_value - value)
        improvement = best_value - average_value

        # self.policy_switch[x] -= 0.25
        # self.policy_switch[x, a] += 1
        
        self.policy_switch[x] += self.learning_rate * (improvement - self.policy_switch[x])
        # print(self.policy_switch[x])
        # if prior_best_a == best_a:
        #     self.policy_switch[x] += 1
        # else:
        #     self.policy_switch[x] -= 1


        
        self.average_learning[x] = (1 - self.learning_rate) * self.average_learning[x] + improvement * self.learning_rate

        if not terminal:
            # if not skip_condition:
            if skip_condition:
                # ap = np.argmax(self.slow_behaviour_learner.get_action_values(xp))
                
                if xp % self.env.size == self.env.size // 2:
                    ap = 0
                elif xp % self.env.size == 0:
                    ap = 2
                elif xp % self.env.size == self.env.size - 1:
                    ap = 2 
                elif xp < self.env.size and xp < self.env.size // 2:
                    ap = 3
                elif xp < self.env.size and xp > self.env.size // 2:
                    ap = 1
                else:
                    raise ValueError(f'xp is {xp} sp: {sp}')
            else:
                ap = self.selectAction(sp)
            # else:
                # if xp in [10, 17, 24, 31, 38, 45]:
                #     ap = 0
                # elif xp in [0, 7, 14, 21, 28, 35]:
                #     ap = 2
                # elif xp in [6, 13, 20, 27, 34, 41]:
                #     ap = 2  
                # else: 
                # policy = self.average_policy[xp, :] / np.linalg.norm(self.average_policy[xp, :], ord=1)
                # ap = self.random.choice(self.num_actions, p=policy)
                # print('this shouldnt happen')
        else:
            ap = None

        if not skip_condition:
            self.prior_x = xp
            self.prior_a = ap

            # print(f'{self.prior_x} a: {self.prior_a}')

        self.cumulative_reward += r
        if globals.blackboard['num_steps_passed'] % globals.blackboard['step_logging_interval'] == 0:
            globals.collector.collect('Q', np.copy(self.behaviour_learner.Q))
            globals.collector.collect('slow_Q', np.copy(self.slow_behaviour_learner.Q))
            globals.collector.collect('policy_agreement', np.copy(self.policy_agreement))
            globals.collector.collect('reward', np.copy(self.cumulative_reward))

            # print(self.cumulative_reward)
            self.cumulative_reward = 0

        return ap

    def agent_end(self, s, a, r, gamma):
        self.update(s, a, None, r, gamma, terminal=True)
        self.behaviour_learner.episode_end()

        self.interim_r = 0
        self.prior_x = None

        # Logging state visitation
        globals.collector.collect('state_visitation', np.copy(self.state_visitations))   
        globals.collector.collect('skipped_states', np.copy(self.skipped_states)) 
        globals.collector.collect('policy_switch', np.copy(self.policy_switch))   
        self.state_visitations[:] = 0
        self.skipped_states[:] = 0
