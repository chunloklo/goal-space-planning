from src.agents.components.learners import ESarsaLambda, QLearner
from src.agents.components.search_control import ActionModelSearchControl_Tabular
from PyExpUtils.utils.random import argmax, choice
from PyFixedReps.BaseRepresentation import BaseRepresentation
from PyFixedReps.Tabular import Tabular
from src.agents.components.approximators import DictModel
from src.agents.components.models import ActionModel_Linear, OptionActionModel_ESARSA, OptionActionModel_Sutton_Tabular, OptionModel_TB_Tabular, OptionModel_Sutton_Tabular
from src.utils import globals
from src.utils import options, param_utils
from src.utils import rlglue
from typing import Dict, Union, Tuple, Any, TYPE_CHECKING
import numpy as np
import numpy.typing as npt
import random

if TYPE_CHECKING:
    # Important for forward reference
    from src.problems.BaseProblem import BaseProblem

class DynaOptions_Tab:
    def __init__(self, problem: 'BaseProblem'):
        self.wrapper_class = rlglue.OptionOneStepWrapper
        self.env = problem.getEnvironment()
        self.representation: Tabular = problem.get_representation('Tabular')
        assert isinstance(self.representation, Tabular)
        self.features = self.representation.features
        self.num_actions = problem.actions
        self.actions = list(range(self.num_actions))
        self.params = globals.param
        self.num_states = self.env.nS
        self.options = problem.options
        self.num_options = len(problem.options)
        self.random = np.random.RandomState(problem.seed)

        # define parameter contract
        params = self.params
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']
        self.planning_steps = params['planning_steps']
        self.model_planning_steps = params['model_planning_steps']
        self.kappa = params['kappa']
        self.behaviour_alg = param_utils.parse_param(params, 'behaviour_alg', lambda p : p in ['QLearner', 'ESarsaLambda']) 
        self.learn_model =  param_utils.parse_param(params, 'learn_model', lambda p : isinstance(p, bool))

        self.search_control = ActionModelSearchControl_Tabular(self.random)
        
        self.goals = problem.get_goals()
        self.goals = [self.representation.encode(goal) for goal in self.goals]
        self.tau = np.zeros(len(self.goals))

        # Loading the pretrained model rather than learning from scratch.
        if not self.learn_model:
            self.option_model, self.option_action_model = problem.get_pretrained_option_model()
        else:
            self.option_model = OptionModel_Sutton_Tabular(self.num_states + 1, self.num_actions, self.num_options, self.options)
            self.option_action_model = OptionActionModel_ESARSA(self.num_states + 1, self.num_actions, self.num_options, self.options)

        # Creating models for actions and options
        self.history_dict = DictModel()

        self.action_model = ActionModel_Linear(self.num_states + 1, self.num_actions)
        
        
        if not self.learn_model:
            transitions = problem.get_all_transitions()

            # Prefilling dict buffer so no exploration is needed
            for t in transitions:
                # t: s, a, sp, reward. gamma
                self.history_dict.update(self.representation.encode(t[0]), t[1], self.representation.encode(t[2]), t[3], t[4])
                # do a complete update with each transition
                self.action_model.update(self.representation.encode(t[0]), t[1], self.representation.encode(t[2]), t[3], t[4], 1)
        

        # +1 accounts for the terminal state
        if self.behaviour_alg == 'QLearner':
            self.behaviour_learner = QLearner(self.num_states + 1, self.num_actions + self.num_options)
        elif self.behaviour_alg == 'ESarsaLambda':
            self.lmbda = params['lambda']
            self.behaviour_learner = ESarsaLambda(self.num_states + 1, self.num_actions + self.num_options)
        else:
            raise NotImplementedError(f'behaviour_alg for {self.behaviour_alg} is not implemented')

        # For logging state visitation
        self.cumulative_reward = 0

    def FA(self):
        return "Tabular"

    def __str__(self):
        return "DynaOptions_Tab"

    def get_policy(self, x: int) -> npt.ArrayLike:
        probs = np.zeros(self.num_actions + self.num_options)
        probs += self.epsilon / (self.num_actions + self.num_options)
        o = np.argmax(self.behaviour_learner.get_action_values(x))
        probs[o] += 1 - self.epsilon
        return probs

    # public method for rlglue
    def selectAction(self, s: Any) -> Tuple[int, int] :
        x = self.representation.encode(s)
        o = self.random.choice(self.num_actions + self.num_options, p = self.get_policy(x))

        if o >= self.num_actions:
            a, _ = options.get_option_info(x, options.from_action_to_option_index(o, self.num_actions), self.options)
        else:
            a=o

        return o,a

    def update(self, s: Any, o, a, sp: Any, r, gamma, terminal: bool = False):
        x = self.representation.encode(s)
        # Treating the terminal state as an additional state in the tabular setting
        xp = self.representation.encode(sp) if not terminal else self.num_states

        # Exploration bonus tracking
        if not globals.blackboard['in_exploration_phase']:
            self.tau += 1
        self.tau[xp == np.array(self.goals)] = 0

        if isinstance(self.behaviour_learner, QLearner):
            self.behaviour_learner.update(x, o, xp, r, gamma, self.alpha)
        elif isinstance(self.behaviour_learner, ESarsaLambda):
            self.behaviour_learner.update(x, o, xp, r, self.get_policy(xp), gamma, self.lmbda, self.alpha)
        else:
            raise NotImplementedError()

        # Updating search control. Order is important.
        self.search_control.update(x, xp)

        self.update_model(x,a,xp,r,gamma)  
        self.planning_step(x, xp)
    
        # not strictly needed because the option action pair shouldn't be used in termination,
        # but it prevents some unneeded computation that could error out with weird indexing.
        if not terminal:
            oa_pair = self.selectAction(sp)
        else:
            # the oa pair doesn't matter if the agent arrived in the terminal state.
            oa_pair = None


        # Logging
        self.cumulative_reward += r
        if globals.blackboard['num_steps_passed'] % globals.blackboard['step_logging_interval'] == 0:
            globals.collector.collect('Q', np.copy(self.behaviour_learner.Q)) 

            # globals.collector.collect('tau', np.copy(self.tau)) 
            globals.collector.collect('reward_rate', np.copy(self.cumulative_reward) / globals.blackboard['step_logging_interval'])
            # print(f'reward rate: {np.copy(self.cumulative_reward) / globals.blackboard["step_logging_interval"]}')
            self.cumulative_reward = 0

        return oa_pair
    
    def option_model_planning_update(self, x, a, xp ,r, gamma):
        if isinstance(self.option_model, OptionModel_Sutton_Tabular):
            self.option_model.update(x, a, xp, r, gamma, self.alpha)
        else:
            raise NotImplementedError(f'Planning update for {type(self.option_model)} is not implemented')

    def update_model(self, x, a, xp, r, gamma):
        """updates the model 
        
        Returns:
            Nothing
        """
        self.history_dict.update(x, a, xp, r, gamma)
        self.action_model.update(x, a, xp, r, gamma, self.alpha)
        if isinstance(self.option_model, OptionModel_Sutton_Tabular):
            self.option_model.update(x, a, xp, r, gamma, self.alpha)
            self.option_action_model.update(x, a, xp, r, gamma, self.alpha)
        else:
            raise NotImplementedError(f'Update for {type(self.option_model)} is not implemented')

    def _planning_update(self, x: int, o: int):
        if (o < self.num_actions):
            # Generating the experience from the action_model
            r, discount, transition_prob = self.action_model.predict(x, o)
            xp = np.argmax(transition_prob)
        else:
            # Generating the experience from the option model
            option_index = options.from_action_to_option_index(o, self.num_actions)
            r, discount, transition_prob = self.option_model.predict(x, option_index)
            transition_prob = np.clip(transition_prob, a_min = 0, a_max = None)
            norm = np.linalg.norm(transition_prob, ord=1)
            if (norm != 0):
                prob = transition_prob / norm
                # +1 here accounts for the terminal state
                xp = self.random.choice(self.num_states + 1, p=prob)
            else:
                xp = None

        # Exploration bonus for +
        r += self.kappa * np.sqrt(np.sum(self.tau[xp == np.array(self.goals)]))

        # xp could be none if the transition probability errored out
        if xp != None:
            if isinstance(self.behaviour_learner, QLearner):
                self.behaviour_learner.planning_update(x, o, xp, r, discount, self.alpha)
            elif isinstance(self.behaviour_learner, ESarsaLambda):
                self.behaviour_learner.planning_update(x, o, xp, r, self.get_policy(xp), discount, self.alpha)
            else:
                raise NotImplementedError()

    def planning_step(self, x:int, xp: int):
        """performs planning, i.e. indirect RL.

        Returns:
            Nothing
        """

        sample_states = self.search_control.sample_states(self.planning_steps, self.history_dict, x, xp)

        for i in range(self.planning_steps):
            plan_x = sample_states[i]

            # Pick a random action/option within all eligable action/options
            # I think there should be a better way of doing this...
            visited_actions = self.history_dict.visited_actions(plan_x)
            action_consistent_options = options.get_action_consistent_options(plan_x, visited_actions, self.options, convert_to_actions=True, num_actions=self.num_actions)
            available_actions = visited_actions + action_consistent_options
            
            a = self.random.choice(available_actions)
            # for a in available_actions: 
            self._planning_update(plan_x, a)

    def agent_end(self, s, o, a, r, gamma):
        self.update(s, o, a, None, r, gamma, terminal=True)
        self.behaviour_learner.episode_end()
        self.option_model.episode_end()
        self.option_action_model.episode_end()
