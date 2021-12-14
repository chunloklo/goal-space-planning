
import numpy as np
from PyExpUtils.utils.random import argmax, choice
import random
from agents.components.learners import ESarsaLambda, QLearner, QLearner_ImageNN
from agents.components.search_control import ActionModelSearchControl_Tabular
from environments.GrazingWorldAdam import GrazingWorldAdamImageFeature
from src.utils import rlglue
from src.utils import globals
from src.utils import options, param_utils
from src.agents.components.models import OptionModel_TB_Tabular, OptionModel_Sutton_Tabular
from src.agents.components.approximators import DictModel
from PyFixedReps.BaseRepresentation import BaseRepresentation
import numpy.typing as npt
from PyFixedReps.Tabular import Tabular
from typing import Dict, Union, Tuple, Any, TYPE_CHECKING
import jax.numpy as jnp
from src.agents.components.buffers import Buffer, DictBuffer
from collections import namedtuple
from src.environments.GrazingWorldAdam import get_all_transitions

if TYPE_CHECKING:
    # Important for forward reference
    from src.problems.BaseProblem import BaseProblem

class DynaOptions_NN:
    def __init__(self, problem: 'BaseProblem'):
        self.wrapper_class = rlglue.OptionOneStepWrapper
        self.env = problem.getEnvironment()
        self.image_representation: GrazingWorldAdamImageFeature = problem.get_representation('Image')
        self.tabular_representation: Tabular = problem.get_representation('Tabular')

        self.num_actions = problem.actions
        self.params = problem.params
        self.num_states = self.env.nS
        self.options = problem.options
        self.num_options = len(problem.options)
        self.random = np.random.RandomState(problem.seed)

        self.behaviour_learner = QLearner_ImageNN(self.image_representation.features(), self.num_actions, 1e-3)
        # # define parameter contract
        params = self.params
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']
        self.polyak_stepsize = params['polyak_stepsize']

        # Replay Buffer:
        self.buffer_size = 100000
        self.batch_size = 8
        self.buffer = Buffer(self.buffer_size, {'s': (2,), 'x': (8, 12, 1), 'a': (), 'xp': (8, 12, 1), 'r': (), 'gamma': ()}, self.random.randint(0,2**31))

        self.planning_steps = param_utils.check_valid(params['planning_steps'], lambda x: isinstance(x, int) and x > 0)
        # self.model_planning_steps = params['model_planning_steps']
        self.kappa = params['kappa']
        # The number of steps that the goal state is not visited before the kappa exploration bonus will be applied to that experience.
        self.kappa_interval = params['kappa_interval']
        self.priority_alpha = params['priority_alpha']
        # self.lmbda = params['lambda']
        # self.model_alg =  param_utils.parse_param(params, 'option_model_alg', lambda p : p in ['sutton'])
        # self.behaviour_alg = param_utils.parse_param(params, 'behaviour_alg', lambda p : p in ['QLearner', 'ESarsaLambda']) 

        # search_control_type = param_utils.parse_param(params, 'search_control', lambda p : p in ['random', 'current', 'td', 'close'])
        # self.search_control = ActionModelSearchControl_Tabular(search_control_type, self.random)
        
        # # DO WE NEED THIS?!?!? CAN WE MAKE THIS SOMEWHERE ELSE?
        self.tau = np.zeros((self.num_options))
        # self.a = -1
        # self.x = -1
        # # Creating models for actions and options
        self.action_model = DictModel()

        self.dict_buffer = DictBuffer()
        transitions = get_all_transitions()
        
        # Prefilling dict buffer so no exploration is needed
        DictBufferData = namedtuple("DictBufferData", "s a sp r gamma")
        for t in transitions:
            self.dict_buffer.update(DictBufferData(t[0], t[1], t[2], t[3], t[4]))
            self.action_model.update(t[0], t[1], t[2], t[3], t[4])

        # For logging state visitation
        self.state_visitations = np.zeros(self.num_states)

        # Logging update distribution
        self.update_distribution = np.zeros(self.num_states)

    def FA(self):
        return "Neural Network"

    def __str__(self):
        return "DynaOptions_NN"

    # def get_policy(self, x: int) -> npt.ArrayLike:
    #     probs = np.zeros(self.num_actions + self.num_options)
    #     probs += self.epsilon / (self.num_actions + self.num_options)
    #     o = np.argmax(self.behaviour_learner.get_action_values(x))
    #     probs[o] += 1 - self.epsilon
    #     return probs

    # public method for rlglue
    def selectAction(self, s: Any) -> Tuple[int, int] :
        if self.random.uniform(0, 1) < self.epsilon:
            a = self.random.choice(4)
        else:
            x = jnp.expand_dims(self.image_representation.encode(s), axis=0)
            q_values = self.behaviour_learner.get_action_values(x)
            a = int(np.argmax(q_values))

        # a = self.random.choice(4)
        # print(a)
        return a, a

    def planning_update(self):
        # Hard coding this for GrazingWorldAdam for now
        tau_states = [13,31,81]

        samples = self.dict_buffer.sample(self.random, self.batch_size, alpha=self.priority_alpha)
        batch_x = jnp.array([self.image_representation.encode(sample.s) for sample in samples])
        batch_a = jnp.array([sample.a for sample in samples])
        batch_xp = jnp.array([self.image_representation.encode(sample.sp) for sample in samples])
        batch_gamma = jnp.array([sample.gamma for sample in samples])
        # batch_r = np.array([sample.r for sample in samples])

        batch_r = np.zeros(len(samples))
        for i, sample in enumerate(samples):
            sp, r, gamma = self.action_model.predict(sample.s, sample.a)
            batch_r[i] = r 

        # exploration bonus
        if not globals.blackboard['in_exploration_phase']:
            for i, sample in enumerate(samples):
                tab_x = int(self.tabular_representation.encode(sample.s))
                try:
                    index = tau_states.index(tab_x)
                    if self.tau[index] > self.kappa_interval:
                    # if globals.blackboard['num_steps_passed'] > 15000 and globals.blackboard['num_steps_passed'] < 30000:
                            batch_r[i] += self.kappa
                except ValueError:
                    pass  
            
        # forming batch
        batch = {}
        batch['x'] = batch_x
        batch['a'] = batch_a
        batch['xp'] = batch_xp
        batch['r'] = batch_r
        batch['gamma'] = batch_gamma

        # batch = self.buffer.sample(self.batch_size)

        _, td_errors = self.behaviour_learner.update(batch, self.polyak_stepsize)  
          
        td_errors = np.abs(td_errors)
        # # Updating TD error in the buffer
        for i, sample in enumerate(samples):
            self.dict_buffer.update(sample, td_errors[i])


    def update(self, s: Any, o, a, sp: Any, r, gamma, terminal: bool = False):

        self.action_model.update(tuple(s), a, sp, r, gamma)

        if sp is None:
            assert gamma == 0
            sp = (0, 0)
        else:
            sp = tuple(sp)
                    
        # # # forming batch
        # batch = {'x': jnp.array([self.image_representation.encode(s)]), 'a': jnp.array([a]), 'xp': jnp.array([self.image_representation.encode(sp)]), 'r': jnp.array([r]), 'gamma': jnp.array([gamma])}
        # # indexing into 0 since it is an array
        # delta = np.abs(self.behaviour_learner.get_deltas(batch))[0]

        # DictBufferData = namedtuple("DictBufferData", "s a sp r gamma")
        # self.dict_buffer.update(DictBufferData(tuple(s), a, sp, r, gamma), delta)

        tab_x = self.tabular_representation.encode(s)

        self.state_visitations[tab_x] += 1


        # Hard coding this for GrazingWorldAdam for now
        tau_states = [13,31,81]

        # Updating tau and exploration bonus
        if not globals.blackboard['in_exploration_phase']:
            self.tau += 1

        try:
            index = tau_states.index(tab_x)
            self.tau[index] = 0
            # if not globals.blackboard['in_exploration_phase']:
                # print(tab_x)
        except ValueError:
            pass    

        x = self.image_representation.encode(s)
        # Treating the terminal state as an additional state in the tabular setting
        xp = self.image_representation.encode(sp) if not terminal else jnp.zeros(self.image_representation.features())
    
        buffer_data = {'s': s, 'x': x, 'a': a, 'xp': xp, 'r': r, 'gamma': gamma}
        self.buffer.update(buffer_data)

        # s = self.action_model.visited_states()
        # print(len(s))

        # adding exploration bonus to batch
        # if not globals.blackboard['in_exploration_phase']:
        for _ in range(self.planning_steps):
            self.planning_update()
    
        # not strictly needed because the option action pair shouldn't be used in termination,
        # but it prevents some unneeded computation that could error out with weird indexing.
        if not terminal:
            oa_pair = self.selectAction(sp)
        else:
            # the oa pair doesn't matter if the agent arrived in the terminal state.
            oa_pair = None

        if globals.blackboard['num_steps_passed'] % globals.blackboard['step_logging_interval'] == 0:
            xs = np.zeros((96, 8, 12, 1))
            for r in range(8):
                for c in range(12):
                    s = (r, c)
                    x = self.image_representation.encode(s)
                    xs[r * 12 + c] = x
            q_values = self.behaviour_learner.get_action_values(xs)
            globals.collector.collect('Q', np.copy(q_values)) 

            # Logging state visitation
            globals.collector.collect('state_visitation', np.copy(self.state_visitations))   
            self.state_visitations[:] = 0

            globals.collector.collect('tau', np.copy(self.tau)) 

        return oa_pair

    def agent_end(self, s, o, a, r, gamma):
        self.update(s, o, a, None, r, gamma, terminal=True)
 
        # self.behaviour_learner.episode_end()
        # self.option_model.episode_end()

        globals.collector.collect('update_distribution', np.copy(self.update_distribution))   
        self.update_distribution[:] = 0
