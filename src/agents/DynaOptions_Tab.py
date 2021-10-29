from typing import Dict, Union, Tuple
import numpy as np
import numpy.typing as npt
from PyExpUtils.utils.random import argmax, choice
import random
from PyFixedReps.Tabular import Tabular
from agents.components.learners import ESarsaLambda, QLearner
from src.utils import rlglue
from src.utils import globals
from src.utils import options, param_utils
from src.agents.components.models import OptionModel_TB_Tabular, OptionModel_Sutton_Tabular
from src.agents.components.approximators import DictModel

class DynaOptions_Tab:
    def __init__(self, features: int, actions: int, params: Dict, seed: int, options, env):
        self.wrapper_class = rlglue.OptionOneStepWrapper

        self.env = env
        self.features = features
        self.num_actions = actions
        self.actions = list(range(self.num_actions))
        self.params = params
        self.num_states = self.env.nS
        self.options = options
        self.num_options = len(options)
        self.random = np.random.RandomState(seed)

        # define parameter contract
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']
        self.planning_steps = params['planning_steps']
        self.model_planning_steps = params['model_planning_steps']
        self.gamma = params['gamma']
        self.kappa = params['kappa']
        self.lmbda = params['lambda']
        self.model_alg =  param_utils.parse_param(params, 'option_model_alg', lambda p : p in ['sutton'])
        self.behaviour_alg = param_utils.parse_param(params, 'behaviour_alg', lambda p : p in ['QLearner', 'ESarsaLambda']) 
        self.search_control = param_utils.parse_param(params, 'search_control', lambda p : p in ['random', 'current', 'close'])
        
        # DO WE NEED THIS?!?!? CAN WE MAKE THIS SOMEWHERE ELSE?
        self.tau = np.zeros((self.num_states, self.num_actions + self.num_options))
        self.a = -1
        self.x = -1

        # +1 accounts for the terminal state
        if self.behaviour_alg == 'QLearner':
            self.behaviour_learner = QLearner(self.num_states + 1, self.num_actions + self.num_options)
        elif self.behaviour_alg == 'ESarsaLambda':
            self.behaviour_learner = ESarsaLambda(self.num_states + 1, self.num_actions + self.num_options)
        else:
            raise NotImplementedError(f'behaviour_alg for {self.behaviour_alg} is not implemented')

        # Creating models for actions and options
        self.action_model = DictModel()
        if self.model_alg == 'sutton':
            self.option_model = OptionModel_Sutton_Tabular(self.num_states + 1, self.num_actions, self.num_options, self.options)
        else:
            raise NotImplementedError(f'option_model_alg for {self.model_alg} is not implemented')

        # For 'close' search control
        self.distance_from_goal = {}

        # Temp logging vars
        self.log_action_selected = []

    def FA(self):
        return "Tabular"

    def __str__(self):
        return "DynaQP_OptionIntra_Tab"

    def get_policy(self, x: int) -> npt.ArrayLike:
        probs = np.zeros(self.num_actions + self.num_options)
        probs += self.epsilon / (self.num_actions + self.num_options)
        o = np.argmax(self.behaviour_learner.get_action_values(x))
        probs[o] += 1 - self.epsilon
        return probs

    # public method for rlglue
    def selectAction(self, x: int) -> Tuple[int, int] :
        o = self.random.choice(self.num_actions + self.num_options, p = self.get_policy(x))

        if o >= self.num_actions:
            a, _ = options.get_option_info(x, options.from_action_to_option_index(o, self.num_actions), self.options)
        else:
            a=o

        return o,a

    def update(self, x, o, a, xp, r, gamma):
        self.log_action_selected.append(o)

        # not strictly needed because the option action pair shouldn't be used in termination,
        # but it prevents some unneeded computation that could error out with weird indexing.
        if (xp != options.GRAZING_WORLD_TERMINAL_STATE):
            oa_pair = self.selectAction(xp)
        else:
            # the oa pair doesn't matter if the agent arrived in the terminal state.
            oa_pair = None

        # Exploration bonus tracking
        self.tau += 1
        self.tau[x, o] = 0

        if isinstance(self.behaviour_learner, QLearner):
            self.behaviour_learner.update(x, o, xp, r, gamma, self.alpha)
        elif isinstance(self.behaviour_learner, ESarsaLambda):
            self.behaviour_learner.update(x, o, xp, r, self.get_policy(xp), self.gamma, self.lmbda, self.alpha)
        else:
            raise NotImplementedError()
        self.update_model(x,a,xp,r)  
        self.planning_step(x, xp, self.search_control)

        return oa_pair
    
    def option_model_planning_update(self, x, a, xp ,r):
        if isinstance(self.option_model, OptionModel_Sutton_Tabular):
            self.option_model.update(x, a, xp, r, self.gamma, self.alpha)
        else:
            raise NotImplementedError(f'Planning update for {type(self.option_model)} is not implemented')

    def update_model(self, x, a, xp, r):
        """updates the model 
        
        Returns:
            Nothing
        """
        if self.search_control == "close":
            if x == self.env.state_encoding(self.env.start_state):
                self.distance_from_goal[x] = 1
            if xp != self.termination_state_index and (xp not in self.distance_from_goal or self.distance_from_goal[xp] > self.distance_from_goal[x] +1):
                self.distance_from_goal[xp] = self.distance_from_goal[x] +1

        self.action_model.update(x, a, xp, r)
        if isinstance(self.option_model, OptionModel_Sutton_Tabular):
            self.option_model.update(x, a, xp, r, self.gamma, self.alpha)
        else:
            raise NotImplementedError(f'Update for {type(self.option_model)} is not implemented')

    def _planning_update(self, x: int, o: int):
        if (o < self.num_actions):
            # Generating the experience from the action_model
            xp, r = self.action_model.predict(x, o)
            discount = self.gamma
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
        r += self.kappa * np.sqrt(self.tau[x, o])

        # xp could be none if the transition probability errored out
        if xp != None:
            if isinstance(self.behaviour_learner, QLearner):
                self.behaviour_learner.planning_update(x, o, xp, r, discount, self.alpha)
            elif isinstance(self.behaviour_learner, ESarsaLambda):
                self.behaviour_learner.planning_update(x, o, xp, r, self.get_policy(xp), discount, self.alpha)
            else:
                raise NotImplementedError()

    def planning_step(self, x:int, xp: int, search_control: str):
        """performs planning, i.e. indirect RL.

        Returns:
            Nothing
        """
        
        # Additional model planning steps if we need them. Usually this is set to 0 though.
        for _ in range(self.model_planning_steps):
            # resample the states
            plan_x = choice(np.array(self.action_model.visited_states()), self.random)
            visited_actions = self.action_model.visited_actions(plan_x)

            # Improving option model!
            # We're improving the model a ton here (updating all 4 actions). We could reduce this later but let's keep it high for now?
            # Disabling this since we are testing with traces
            for a in visited_actions:
                xp, r = self.action_model.predict(plan_x, a)
                self.option_model_planning_update(plan_x, a, xp, r)

        if search_control == "close":
            visited_states, distances = [], []
            for k in self.action_model.visited_states():
                visited_states.append(k)
                distances.append(self.distance_from_goal[k])
            normed_distances = [i/sum(distances) for i in distances]

        for _ in range(self.planning_steps):
            if search_control=="random":
                plan_x = choice(np.array(self.action_model.visited_states()), self.random)
            elif search_control =="current":
                visited_states = list(self.action_model.visited_states())
                if (xp in list(visited_states)):
                    plan_x = xp
                else:
                    # Random if you haven't visited the next state yet
                    plan_x = x
            elif search_control =="close":
                plan_x = self.random.choice(np.array(list(visited_states)), p = normed_distances)
            visited_actions = self.action_model.visited_actions(plan_x)
            
            # Pick a random action/option within all eligable action/options
            # I think there should be a better way of doing this...
            action_consistent_options = options.get_action_consistent_options(plan_x, visited_actions, self.options, convert_to_actions=True, num_actions=self.num_actions)
            available_actions = visited_actions + action_consistent_options
            update_options = [choice(np.array(available_actions), self.random)]
            
            for a in update_options: 
                self._planning_update(plan_x, a)

    def agent_end(self, x, o, a, r, gamma):
        self.update(x, o, a, options.GRAZING_WORLD_TERMINAL_STATE, r, gamma)
        self.behaviour_learner.episode_end()
        self.option_model.episode_end()

        globals.collector.collect('action_selected', self.log_action_selected.copy())
        self.log_action_selected = []  