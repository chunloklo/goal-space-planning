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
from src.environments.GrazingWorldAdam import get_pretrained_option_model, state_index_to_coord

if TYPE_CHECKING:
    # Important for forward reference
    from src.problems.BaseProblem import BaseProblem

class Dyna_Tab:
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
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']
        self.planning_steps = params['planning_steps']
        self.gamma = params['gamma']
        self.kappa = params['kappa']
        self.lmbda = params['lambda']
        self.behaviour_alg = param_utils.parse_param(params, 'behaviour_alg', lambda p : p in ['QLearner', 'ESarsaLambda']) 
        search_control_type = param_utils.parse_param(params, 'search_control', lambda p : p in ['random', 'current', 'td', 'close'])
        self.search_control = ActionModelSearchControl_Tabular(search_control_type, self.random)
        
        self.option_alg = param_utils.parse_param(params, 'option_alg', lambda p : p in ['None', 'DecisionTime', 'Background']) 

        # Instantiating option models
        # self.option_model = OptionModel_Sutton_Tabular(self.num_states + 1, self.num_actions, self.num_options, self.options)
        # self.option_action_model = OptionActionModel_Sutton_Tabular(self.num_states + 1, self.num_actions, self.num_options, self.options)

        # Loading the pretrained model rather than learning from scatch.
        # This is specifically for GrazingWorldAdam
        self.option_model, self.option_action_model = get_pretrained_option_model()

        self.tau = np.zeros((self.num_states, self.num_actions))
        self.a = -1
        self.x = -1

        # +1 accounts for the terminal state
        if self.behaviour_alg == 'QLearner':
            self.behaviour_learner = QLearner(self.num_states + 1, self.num_actions)
        elif self.behaviour_alg == 'ESarsaLambda':
            self.behaviour_learner = ESarsaLambda(self.num_states + 1, self.num_actions)
        else:
            raise NotImplementedError(f'behaviour_alg for {self.behaviour_alg} is not implemented')

        # Creating models for actions and options
        self.action_model = DictModel()

        # For logging state visitation
        self.state_visitations = np.zeros(self.num_states)
        self.cumulative_reward = 0

    def FA(self):
        return "Tabular"

    def __str__(self):
        return "Dyna_Tab"

    def get_policy(self, x: int) -> npt.ArrayLike:
        probs = np.zeros(self.num_actions)
        probs += self.epsilon / (self.num_actions)
        a = np.argmax(self.behaviour_learner.get_action_values(x))
        probs[a] += 1 - self.epsilon
        return probs

    # public method for rlglue
    def selectAction(self, s: int) -> Tuple[int, int] :
        x = self.representation.encode(s)
        a = self.random.choice(self.num_actions, p = self.get_policy(x))
        return a

    def update(self, s, a, sp, r, gamma, terminal: bool = False):
        x = self.representation.encode(s)
        # Treating the terminal state as an additional state in the tabular setting
        xp = self.representation.encode(sp) if not terminal else self.num_states

        self.state_visitations[x] += 1

        # Exploration bonus tracking
        if not globals.blackboard['in_exploration_phase']:
            self.tau += 1
        self.tau[x, a] = 0

        if isinstance(self.behaviour_learner, QLearner):
            self.behaviour_learner.update(x, a, xp, r, gamma, self.alpha)
        elif isinstance(self.behaviour_learner, ESarsaLambda):
            self.behaviour_learner.update(x, a, xp, r, self.get_policy(xp), self.gamma, self.lmbda, self.alpha)
        else:
            raise NotImplementedError()

        # Updating search control. Order is important.
        self.search_control.update(x, xp)

        self.update_model(x,a,xp,r, gamma)  
        self.planning_step(x, xp)

        if not terminal:
            ap = self.selectAction(sp)
        else:
            ap = None

        if globals.blackboard['num_steps_passed'] % globals.blackboard['step_logging_interval'] == 0:
            globals.collector.collect('Q', np.copy(self.behaviour_learner.Q)) 

            # Logging state visitation
            globals.collector.collect('state_visitation', np.copy(self.state_visitations))   
            self.state_visitations[:] = 0

            # globals.collector.collect('tau', np.copy(self.tau)) 
            globals.collector.collect('reward_rate', np.copy(self.cumulative_reward) / globals.blackboard['step_logging_interval'])
            # print(f'reward rate: {np.copy(self.cumulative_reward) / globals.blackboard["step_logging_interval"]}')
            self.cumulative_reward = 0


        return ap
    def update_model(self, x, a, xp, r, gamma):
        """updates the model 
        
        Returns:
            Nothing
        """
        self.action_model.update(x, a, xp, r, gamma)
        self.option_model.update(x, a, xp, r, gamma, self.alpha)
        self.option_action_model.update(x, a, xp, r, gamma, self.alpha)

    def _planning_update(self, x: int, a: int):
        xp, r, gamma = self.action_model.predict(x, a)
        discount = gamma

        # Exploration bonus for +
        # These states are specifically for GrazingWorldAdam
        # if x in [13,31,81]:
        #     factor = 1
        # else:
        #     factor = 0.0

        # Disabling exploration bonus for now
        factor = 0.0
        
        r += self.kappa * factor * np.sqrt(self.tau[x, a])

        if isinstance(self.behaviour_learner, QLearner):
            self.behaviour_learner.planning_update(x, a, xp, r, discount, self.alpha)
        elif isinstance(self.behaviour_learner, ESarsaLambda):
            self.behaviour_learner.planning_update(x, a, xp, r, self.get_policy(xp), discount, self.alpha)
        else:
            raise NotImplementedError()

    def _sample_option(self, x, o, a: Optional[int] = None):
        if a == None:
            r, discount, transition_prob = self.option_model.predict(x, o)
        else:
            r, discount, transition_prob = self.option_action_model.predict(x, a, o)
        

        norm = np.linalg.norm(transition_prob, ord=1)
        prob = transition_prob / norm
        # +1 here accounts for the terminal state
        xp = self.random.choice(self.num_states + 1, p=prob)
    
        return r, discount, xp

    def _get_option_values(self, x):
        option_values = np.zeros((self.num_actions, self.num_options))

        for o in range(self.num_options):
            for a in range(self.num_actions):
                r, discount, xp = self._sample_option(x, o, a)
                option_values[a, o] = r + discount * np.max(self.behaviour_learner.get_action_values(xp))
        
        return option_values

    def _option_planning_update(self, x: int):
        option_values = self._get_option_values(x)
        max_option_values = np.max(option_values, axis=1)

        # Updating towards option values
        max_values = np.maximum(max_option_values, self.behaviour_learner.get_action_values(x))
        # [2022-01-11 chunlok] This update is currently a hacky way of doing it that only works with the QLearner I believe.
        self.behaviour_learner.Q[x, :] += self.alpha * (max_values - self.behaviour_learner.Q[x, :])

    def planning_step(self, x:int, xp: int):
        """performs planning, i.e. indirect RL.

        Returns:
            Nothing
        """

        sample_states = self.search_control.sample_states(self.planning_steps, self.action_model, x, xp)

        for i in range(self.planning_steps):
            plan_x = sample_states[i]
            visited_actions = self.action_model.visited_actions(plan_x)
            for a in visited_actions: 
                self._planning_update(plan_x, a)

        # Option planning update
        sample_states = self.search_control.sample_states(self.planning_steps, self.action_model, x, xp)

        if self.option_alg == 'Background':
            for plan_x in sample_states:
                self._option_planning_update(plan_x)


    def agent_end(self, s, a, r, gamma):
        self.update(s, a, None, r, gamma, terminal=True)
        self.behaviour_learner.episode_end()

        # Logging state visitation
        globals.collector.collect('state_visitation', np.copy(self.state_visitations))   
        self.state_visitations[:] = 0
