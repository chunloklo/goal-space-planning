
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
from src.agents.components.buffers import Buffer

if TYPE_CHECKING:
    # Important for forward reference
    from src.problems.BaseProblem import BaseProblem

class DynaOptions_NN:
    def __init__(self, problem: 'BaseProblem'):
        self.wrapper_class = rlglue.OptionOneStepWrapper
        self.env = problem.getEnvironment()
        self.image_representation: GrazingWorldAdamImageFeature = problem.get_representation('Image')
        self.tabular_representation: Tabular = problem.get_representation('Tabular')

        # print(self.representation)

        # self.features = self.representation.features
        self.num_actions = problem.actions
        # self.actions = list(range(self.num_actions))
        self.params = problem.params
        # self.num_states = self.env.nS
        # self.options = problem.options
        # self.num_options = len(problem.options)
        self.random = np.random.RandomState(problem.seed)

        self.behaviour_learner = QLearner_ImageNN(self.image_representation.features(), self.num_actions, 1e-3)
        # # define parameter contract
        params = self.params
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']

        # Replay Buffer:
        self.buffer_size = 1000
        self.batch_size = 8
        self.buffer = Buffer(self.buffer_size, {'x': (8, 12, 1), 'a': (), 'xp': (8, 12, 1), 'r': (), 'gamma': ()}, self.random.randint(0,2**31))

        # self.planning_steps = params['planning_steps']
        # self.model_planning_steps = params['model_planning_steps']
        # self.kappa = params['kappa']
        # self.lmbda = params['lambda']
        # self.model_alg =  param_utils.parse_param(params, 'option_model_alg', lambda p : p in ['sutton'])
        # self.behaviour_alg = param_utils.parse_param(params, 'behaviour_alg', lambda p : p in ['QLearner', 'ESarsaLambda']) 

        # search_control_type = param_utils.parse_param(params, 'search_control', lambda p : p in ['random', 'current', 'td', 'close'])
        # self.search_control = ActionModelSearchControl_Tabular(search_control_type, self.random)
        
        # # DO WE NEED THIS?!?!? CAN WE MAKE THIS SOMEWHERE ELSE?
        # self.tau = np.zeros((self.num_states, self.num_actions + self.num_options))
        # self.a = -1
        # self.x = -1

        # # +1 accounts for the terminal state
        # if self.behaviour_alg == 'QLearner':
        #     self.behaviour_learner = QLearner(self.num_states + 1, self.num_actions + self.num_options)
        # elif self.behaviour_alg == 'ESarsaLambda':
        #     self.behaviour_learner = ESarsaLambda(self.num_states + 1, self.num_actions + self.num_options)
        # else:
        #     raise NotImplementedError(f'behaviour_alg for {self.behaviour_alg} is not implemented')

        # # Creating models for actions and options
        # self.action_model = DictModel()
        # if self.model_alg == 'sutton':
        #     self.option_model = OptionModel_Sutton_Tabular(self.num_states + 1, self.num_actions, self.num_options, self.options)
        # else:
        #     raise NotImplementedError(f'option_model_alg for {self.model_alg} is not implemented')

        # # For logging state visitation
        # self.state_visitations = np.zeros(self.num_states)


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
        x = jnp.expand_dims(self.image_representation.encode(s), axis=0)
        q_values = self.behaviour_learner.get_action_values(x)
        a = int(np.argmax(q_values))

        if self.random.uniform(0, 1) < self.epsilon:
            a = self.random.choice(4)
        # a = self.random.choice(4)
        # print(a)
        return a, a



        # x = self.representation.encode(s)
        # o = self.random.choice(self.num_actions + self.num_options, p = self.get_policy(x))

        # if o >= self.num_actions:
        #     a, _ = options.get_option_info(x, options.from_action_to_option_index(o, self.num_actions), self.options)
        # else:
        #     a=o

        # return o,a

    def update(self, s: Any, o, a, sp: Any, r, gamma, terminal: bool = False):
        x = self.image_representation.encode(s)
        # Treating the terminal state as an additional state in the tabular setting
        xp = self.image_representation.encode(sp) if not terminal else jnp.zeros(self.image_representation.features())

        buffer_data = {'x': x, 'a': a, 'xp': xp, 'r': r, 'gamma': gamma}
        self.buffer.update(buffer_data)

        batch = self.buffer.sample(self.batch_size)

        self.behaviour_learner.update(batch)
        # print('hello1')
        # print(self.behaviour_learner.get_action_values(x))
        # sadasdsd

        # self.state_visitations[x] += 1

        # # Exploration bonus tracking
        # if not globals.blackboard['in_exploration_phase']:
        #     self.tau += 1
        # self.tau[x, o] = 0

        # if isinstance(self.behaviour_learner, QLearner):
        #     self.behaviour_learner.update(x, o, xp, r, gamma, self.alpha)
        # elif isinstance(self.behaviour_learner, ESarsaLambda):
        #     self.behaviour_learner.update(x, o, xp, r, self.get_policy(xp), gamma, self.lmbda, self.alpha)
        # else:
        #     raise NotImplementedError()

        # # Updating search control. Order is important.
        # self.search_control.update(x, xp)

        # self.update_model(x,a,xp,r,gamma)  
        # self.planning_step(x, xp)
    
        # not strictly needed because the option action pair shouldn't be used in termination,
        # but it prevents some unneeded computation that could error out with weird indexing.
        if not terminal:
            oa_pair = self.selectAction(sp)
        else:
            # the oa pair doesn't matter if the agent arrived in the terminal state.
            oa_pair = None

        return oa_pair
    
    # def option_model_planning_update(self, x, a, xp ,r, gamma):
    #     if isinstance(self.option_model, OptionModel_Sutton_Tabular):
    #         self.option_model.update(x, a, xp, r, gamma, self.alpha)
    #     else:
    #         raise NotImplementedError(f'Planning update for {type(self.option_model)} is not implemented')

    # def update_model(self, x, a, xp, r, gamma):
    #     """updates the model 
        
    #     Returns:
    #         Nothing
    #     """
    #     self.action_model.update(x, a, xp, r, gamma)
    #     if isinstance(self.option_model, OptionModel_Sutton_Tabular):
    #         self.option_model.update(x, a, xp, r, gamma, self.alpha)
    #     else:
    #         raise NotImplementedError(f'Update for {type(self.option_model)} is not implemented')

    # def _planning_update(self, x: int, o: int):
    #     if (o < self.num_actions):
    #         # Generating the experience from the action_model
    #         xp, r, gamma = self.action_model.predict(x, o)
    #         discount = gamma
    #     else:
    #         # Generating the experience from the option model
    #         option_index = options.from_action_to_option_index(o, self.num_actions)
    #         r, discount, transition_prob = self.option_model.predict(x, option_index)
    #         transition_prob = np.clip(transition_prob, a_min = 0, a_max = None)
    #         norm = np.linalg.norm(transition_prob, ord=1)
    #         if (norm != 0):
    #             prob = transition_prob / norm
    #             # +1 here accounts for the terminal state
    #             xp = self.random.choice(self.num_states + 1, p=prob)
    #         else:
    #             xp = None

    #     # Exploration bonus for +
    #     # These states are specifically or GrazingWorldAdam
    #     if x in [13,31]:
    #         factor = 1
    #     else:
    #         factor = 0.0
    #     r += self.kappa * factor * np.sqrt(self.tau[x, o])

    #     # xp could be none if the transition probability errored out
    #     if xp != None:
    #         if isinstance(self.behaviour_learner, QLearner):
    #             self.behaviour_learner.planning_update(x, o, xp, r, discount, self.alpha)
    #         elif isinstance(self.behaviour_learner, ESarsaLambda):
    #             self.behaviour_learner.planning_update(x, o, xp, r, self.get_policy(xp), discount, self.alpha)
    #         else:
    #             raise NotImplementedError()

    # def planning_step(self, x:int, xp: int):
    #     """performs planning, i.e. indirect RL.

    #     Returns:
    #         Nothing
    #     """
        
    #     # Additional model planning steps if we need them. Usually this is set to 0 though.
    #     for _ in range(self.model_planning_steps):
    #         # resample the states
    #         plan_x = choice(np.array(self.action_model.visited_states()), self.random)
    #         visited_actions = self.action_model.visited_actions(plan_x)

    #         # Improving option model!
    #         # We're improving the model a ton here (updating all 4 actions). We could reduce this later but let's keep it high for now?
    #         # Disabling this since we are testing with traces
    #         for a in visited_actions:
    #             xp, r, gamma = self.action_model.predict(plan_x, a)
    #             self.option_model_planning_update(plan_x, a, xp, r, gamma)

    #     sample_states = self.search_control.sample_states(self.planning_steps, self.action_model, x, xp)

    #     for i in range(self.planning_steps):
    #         plan_x = sample_states[i]

    #         # Pick a random action/option within all eligable action/options
    #         # I think there should be a better way of doing this...
    #         visited_actions = self.action_model.visited_actions(plan_x)
    #         action_consistent_options = options.get_action_consistent_options(plan_x, visited_actions, self.options, convert_to_actions=True, num_actions=self.num_actions)
    #         available_actions = visited_actions + action_consistent_options
            
    #         for a in available_actions: 
    #             self._planning_update(plan_x, a)

    def agent_end(self, s, o, a, r, gamma):
        self.update(s, o, a, None, r, gamma, terminal=True)

        xs = np.zeros((96, 8, 12, 1))
        for r in range(8):
            for c in range(12):
                s = (r, c)
                x = self.image_representation.encode(s)
                xs[r * 12 + c] = x
        q_values = self.behaviour_learner.get_action_values(xs)
        globals.collector.collect('Q', np.copy(q_values))        
        # self.behaviour_learner.episode_end()
        # self.option_model.episode_end()

        # Logging state visitation
        # globals.collector.collect('state_visitation', np.copy(self.state_visitations))   
        # self.state_visitations[:] = 0
