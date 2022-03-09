
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
from typing import Dict, Optional, Union, Tuple, Any, TYPE_CHECKING
import jax.numpy as jnp
from src.agents.components.buffers import Buffer, DictBuffer
from collections import namedtuple
from src.environments.GrazingWorldAdam import get_all_transitions

if TYPE_CHECKING:
    # Important for forward reference
    from src.problems.BaseProblem import BaseProblem

# [2022-03-09 chunlok] Copied over from DynaOptions_NN. For now, getting this to work with Pinball env.
class Dyna_NN:
    def __init__(self, problem: 'BaseProblem'):
        self.wrapper_class = rlglue.OneStepWrapper
        self.env = problem.getEnvironment()
        # self.image_representation: GrazingWorldAdamImageFeature = problem.get_representation('Image')
        # self.tabular_representation: Tabular = problem.get_representation('Tabular')

        self.num_actions = problem.actions
        self.params = problem.params
        # self.num_states = self.env.shape[0] * self.env.shape[1]
        # self.options = problem.options
        # self.num_options = len(problem.options)
        self.random = np.random.RandomState(problem.seed)

        # # define parameter contract
        params = self.params
        # self.alpha = params['alpha']
        # self.epsilon = params['epsilon']
        # self.polyak_stepsize = params['polyak_stepsize']
        # self.planning_steps = param_utils.check_valid(params['planning_steps'], lambda x: isinstance(x, int) and x > 0)
        # # Bonus reward for exploration
        # self.kappa = params['kappa']
        # # The number of steps that the goal state is not visited before the kappa exploration bonus will be applied to that experience.
        # self.kappa_interval = params['kappa_interval']

        # self.behaviour_learner = QLearner_ImageNN(self.image_representation.features(), self.num_actions, self.alpha)
        # self.batch_size = params['batch_size']

        # # Replay Buffer:
        # # self.buffer_size = 100000
        # # self.buffer = Buffer(self.buffer_size, {'s': (2,), 'x': (8, 12, 1), 'a': (), 'xp': (8, 12, 1), 'r': (), 'gamma': ()}, self.random.randint(0,2**31))

        # # self.model_planning_steps = params['model_planning_steps']

        # self.priority_alpha = params['priority_alpha']
        # self.option_value_alg = param_utils.parse_param(params, 'option_value_alg', lambda p : p in ['None', 'action_selection', 'bootstrap', 'base_target', 'value_shift'])


        # self.tau = np.zeros((self.num_options))
        # # # Creating models for actions and options
        # self.action_model = DictModel()

        # self.dict_buffer = DictBuffer()
        # transitions = get_all_transitions()
        
        # # Prefilling dict buffer so no exploration is needed
        # DictBufferData = namedtuple("DictBufferData", "s a sp r gamma")
        # for t in transitions:
        #     self.dict_buffer.update(DictBufferData(t[0], t[1], t[2], t[3], t[4]))
        #     self.action_model.update(t[0], t[1], t[2], t[3], t[4])

        # self.option_model, self.action_option_model = get_pretrained_option_model()

        # # For logging state visitation
        # self.state_visitations = np.zeros(self.num_states)

        # # Logging update distribution
        # self.update_distribution = np.zeros(self.num_states)

        # self.cumulative_reward = 0

    def FA(self):
        return "Neural Network"

    def __str__(self):
        return "Dyna_NN"


    # public method for rlglue
        # public method for rlglue
    def selectAction(self, s: Any) -> int:
        return self.random.choice(self.num_actions)
        # x = self.representation.encode(s)

        # if self.use_optimal_options:
        #     # UP = 0
        #     # RIGHT = 1
        #     # DOWN = 2
        #     # LEFT = 3

        #     if s[1] == 0 or s[1] == self.env.size - 1:
        #         return 2

        #     if s[1] == (self.env.size - 1) // 2 and s[0] != 0:
        #         return 0

        #     return 0

        # # if not self.skip_action:
        # return self.random.choice(self.num_actions, p = self.get_policy(x))


    # def planning_update(self):
    #     # Hard coding this for GrazingWorldAdam for now
    #     tau_states = [13,31,81]

    #     samples = self.dict_buffer.sample(self.random, self.batch_size, alpha=self.priority_alpha)
    #     batch_x = jnp.array([self.image_representation.encode(sample.s) for sample in samples])
    #     batch_a = jnp.array([sample.a for sample in samples])
    #     batch_xp = jnp.array([self.image_representation.encode(sample.sp) for sample in samples])
    #     batch_gamma = jnp.array([sample.gamma for sample in samples])
    #     # batch_r = np.array([sample.r for sample in samples])

    #     batch_r = np.zeros(len(samples))
    #     batch_option_values = np.zeros((len(samples), self.num_options))
    #     for i, sample in enumerate(samples):
    #         sp, r, gamma = self.action_model.predict(sample.s, sample.a)
    #         batch_r[i] = r 


    #     # exploration bonus
    #     if not globals.blackboard['in_exploration_phase']:
    #         for i, sample in enumerate(samples):
    #             tab_x = int(self.tabular_representation.encode(sample.s))
    #             try:
    #                 index = tau_states.index(tab_x)
    #                 if self.tau[index] > self.kappa_interval:
    #                 # if globals.blackboard['num_steps_passed'] > 15000 and globals.blackboard['num_steps_passed'] < 30000:
    #                         batch_r[i] += self.kappa
    #             except ValueError:
    #                 pass  
            
    #     # forming batch
    #     batch = {}
    #     batch['x'] = batch_x
    #     batch['a'] = batch_a
    #     batch['xp'] = batch_xp
    #     batch['r'] = batch_r
    #     batch['gamma'] = batch_gamma

    #     update_type = 'None'

    #     if self.option_value_alg == 'base_target':
    #         update_type = 'base_target'

    #         batch['best_option_q_sa'] = np.zeros((len(samples)))
    #         for i, sample in enumerate(samples):
    #             batch['best_option_q_sa'][i] = np.max(self._get_option_values(sample.s, sample.a)[1])
        
    #     if self.option_value_alg == 'bootstrap':
    #         update_type = 'bootstrap'

    #         batch['best_option_v_sp'] = np.zeros((len(samples)))
    #         for i, sample in enumerate(samples):
    #             batch['best_option_v_sp'][i] = np.max(self._get_option_values(sample.sp)[1])

        
    #     _, td_errors = self.behaviour_learner.update(batch, self.polyak_stepsize, update_type=update_type)  
          
    #     td_errors = np.abs(td_errors)
    #     # # Updating TD error in the buffer
    #     for i, sample in enumerate(samples):
    #         self.dict_buffer.update(sample, td_errors[i])

    #     if self.option_value_alg == 'value_shift':
    #         samples = self.dict_buffer.sample(self.random, 8, alpha=0)

    #         batch = {}
    #         batch['x'] = jnp.array([self.image_representation.encode(sample.s) for sample in samples])
    #         batch['s'] = [sample.s for sample in samples]

    #         best_option_values = np.max(self._get_batch_option_values(batch), axis=1)

    #         # print(best_option_values)
    #         batch['best_option_values'] = best_option_values
    #         self.behaviour_learner.shift_update(batch)  

    #         # # print(option_values)
    #         # print(best_option_values.shape)
    #         # sdsad
    #         # for i, sample in enumerate(samples):
#         #     for a in range(self.num_actions):
    #         #         _, option_qs = self._get_option_values(sample.s, a)
    #         #         option_values[i, a] = np.max(option_qs)

    #         #     # print(np.maximum(action_values, option_values))
                
    #         # option_action_values = np.maximum(action_values, option_values)
    #         # batch['values'] = option_action_values
    #         # self.behaviour_learner.shift_update(batch)  
    #         # batch_r = np.array([sample.r for sample in samples])

    #         # Get option values
    #         # Pass them off to learner to perform things
    #         pass


    def update(self, s: Any, a, sp: Any, r, gamma, terminal: bool = False):
        print(s, a, sp, gamma, terminal)
        # print(a)
        # print(gamma)
        # print(s)
        # print(sp)
        # print(terminal)
        # print(gamma)
        # self.action_model.update(tuple(s), a, sp, r, gamma)

        # if sp is None:
        #     assert gamma == 0
        #     sp = (0, 0)
        # else:
        #     sp = tuple(sp)
                    
        # # # forming batch
        # batch = {'x': jnp.array([self.image_representation.encode(s)]), 'a': jnp.array([a]), 'xp': jnp.array([self.image_representation.encode(sp)]), 'r': jnp.array([r]), 'gamma': jnp.array([gamma])}
        # # indexing into 0 since it is an array
        # delta = np.abs(self.behaviour_learner.get_deltas(batch))[0]

        # DictBufferData = namedtuple("DictBufferData", "s a sp r gamma")
        # self.dict_buffer.update(DictBufferData(tuple(s), a, sp, r, gamma), delta)

        # tab_x = self.tabular_representation.encode(s)

        # self.state_visitations[tab_x] += 1


        # # Hard coding this for GrazingWorldAdam for now
        # tau_states = [13,31,81]

        # # Updating tau and exploration bonus
        # if not globals.blackboard['in_exploration_phase']:
        #     self.tau += 1

        # try:
        #     index = tau_states.index(tab_x)
        #     self.tau[index] = 0
        #     # if not globals.blackboard['in_exploration_phase']:
        #         # print(tab_x)
        # except ValueError:
        #     pass    

        # x = self.image_representation.encode(s)
        # # Treating the terminal state as an additional state in the tabular setting
        # xp = self.image_representation.encode(sp) if not terminal else jnp.zeros(self.image_representation.features())
    
        # # buffer_data = {'s': s, 'x': x, 'a': a, 'xp': xp, 'r': r, 'gamma': gamma}
        # # self.buffer.update(buffer_data)

        # # s = self.action_model.visited_states()
        # # print(len(s))

        # # adding exploration bonus to batch
        # # if not globals.blackboard['in_exploration_phase']:
        # for _ in range(self.planning_steps):
        #     self.planning_update()



        # # Logging
        # self.cumulative_reward += r
        # if globals.blackboard['num_steps_passed'] % globals.blackboard['step_logging_interval'] == 0:
        #     xs = np.zeros((96, 8, 12, 1))
        #     for r in range(8):
        #         for c in range(12):
        #             s = (r, c)
        #             x = self.image_representation.encode(s)
        #             xs[r * 12 + c] = x
        #     q_values = self.behaviour_learner.get_action_values(xs)
        #     globals.collector.collect('Q', np.copy(q_values)) 

        #     # Logging state visitation
        #     globals.collector.collect('state_visitation', np.copy(self.state_visitations))   
        #     self.state_visitations[:] = 0

        #     globals.collector.collect('tau', np.copy(self.tau)) 
        #     globals.collector.collect('reward_rate', np.copy(self.cumulative_reward) / globals.blackboard['step_logging_interval'])
        #     # print(f'reward rate: {np.copy(self.cumulative_reward) / globals.blackboard["step_logging_interval"]}')
        #     self.cumulative_reward = 0

        if not terminal:
            ap = self.selectAction(sp)
        else:
            ap = None


        return ap

    def agent_end(self, s, a, r, gamma):
        self.update(s, a, None, r, gamma, terminal=True)
 
        # self.behaviour_learner.episode_end()
        # self.option_model.episode_end()