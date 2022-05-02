
import numpy as np
from PyExpUtils.utils.random import argmax, choice
import random
from src.agents.components.EQRC_NN import EQRC_NN
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
from .components.QLearner_NN import QLearner_NN
import cloudpickle

from src.utils.log_utils import get_last_pinball_action_value_map, run_if_should_log

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
        
        params = self.params
        self.step_size = param_utils.parse_param(params, 'step_size', lambda p : isinstance(p, float) and p >= 0)
        self.batch_size = param_utils.parse_param(params, 'batch_size', lambda p : isinstance(p, int) and p > 0)
        self.epsilon = param_utils.parse_param(params, 'step_size', lambda p : isinstance(p, float) and p >= 0)
        self.batch_num = param_utils.parse_param(params, 'batch_num', lambda p : isinstance(p, int) and p > 0, optional=True, default=1)
        self.behaviour_alg = param_utils.parse_param(params, 'behaviour_alg', lambda p : p in ['DQN', 'QRC'])

        if self.behaviour_alg == 'QRC':
            self.beta = param_utils.parse_param(params, 'beta', lambda p : isinstance(p, float), optional=True, default=1.0)
            self.behaviour_learner = EQRC_NN((4,), self.num_actions, self.step_size, self.epsilon, beta=self.beta)
        elif self.behaviour_alg == 'DQN':
            self.polyak_stepsize = param_utils.parse_param(params, 'polyak_stepsize', lambda p : isinstance(p, float) and p >= 0)
            self.behaviour_learner = QLearner_NN((4,), 5, self.step_size, self.polyak_stepsize, 0.0)
        
        self.goals = problem.goals
        self.num_goals = self.goals.num_goals

        self.buffer_size = 1000000
        self.min_buffer_size_before_update = 10000
        self.buffer = Buffer(self.buffer_size, 
            {'x': (4,), 'a': (), 'xp': (4,), 'r': (), 'gamma': ()}, 
            self.random.randint(0,2**31),
            {'a': np.int32})
        self.num_steps_in_ep = 0

        # Name of the file that contains the pretrained behavior that the agent should load. If None, its starts from scratch
        self.load_behaviour_name = param_utils.parse_param(params, 'load_behaviour_name', lambda p : isinstance(p, str) or p is None, optional=True, default=None)
        if self.load_behaviour_name is not None:
            agent = pickle.load(open(f'src/environments/data/pinball/{self.load_behaviour_name}_agent.pkl', 'rb'))
            self.behaviour_learner = agent.behaviour_learner
            self.buffer = agent.buffer

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
        for _ in range(self.batch_num):
            data = self.buffer.sample(self.batch_size)
            self.behaviour_learner.update(data)

    def update(self, s: Any, a, sp: Any, r, gamma, info, terminal: bool = False):

        self.buffer.update({'x': s, 'a': a, 'xp': sp, 'r': r, 'gamma': gamma})
        self.num_steps_in_ep += 1
        
        if r == 10000:
            self.num_term += 1
            if globals.aux_config.get('show_progress'):
                print(f'terminated! term_number: {self.num_term} {self.num_steps_in_ep}')
            globals.collector.collect('num_steps_in_ep', self.num_steps_in_ep)
            self.num_steps_in_ep = 0

        self.update_behavior()

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

    def agent_end(self, s, a, r, gamma, info):
        self.update(s, a, s, r, 0, info, terminal=True)
        # self.behaviour_learner.episode_end()
        # self.option_model.episode_end()

    def experiment_end(self):
        # Saving the agent goal learners
        save_behaviour_name = param_utils.parse_param(self.params, 'save_behaviour_name', lambda p: isinstance(p, str) or p is None, optional=True, default=None)
        if save_behaviour_name:
            cloudpickle.dump(self, open(f'./src/environments/data/pinball/{save_behaviour_name}_agent.pkl', 'wb'))

        q_map = get_last_pinball_action_value_map(1, self.behaviour_learner.get_action_values)
        globals.collector.collect('q_map', q_map[0])