from typing import Dict, Union, Tuple
import numpy as np
import numpy.typing as npt
from PyExpUtils.utils.random import argmax, choice
import random
from src.utils import rlglue, globals, options, param_utils
from src.agents.components.approximators import DictModel
from src.agents.components.models import OptionModel_Sutton_Tabular
from src.agents.components.learners import QLearner, ESarsaLambda

# Dyna but instead of doing Q Planning we do option planning
class OptionPlanning_Tab:
    def __init__(self, features: int, actions: int, params: Dict, seed: int, options, env):
        self.wrapper_class = rlglue.OneStepWrapper

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
        self.search_control = param_utils.parse_param(params, 'search_control', lambda p : p in ['random', 'current'])
        
        self.tau = np.zeros((self.num_states, self.num_actions + self.num_options))
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
        if self.model_alg == 'sutton':
            self.option_model = OptionModel_Sutton_Tabular(self.num_states + 1, self.num_actions, self.num_options, self.options)
        else:
            raise NotImplementedError(f'option_model_alg for {self.model_alg} is not implemented')

    def FA(self):
        return "Tabular"

    def __str__(self):
        return "OptionPlanning_Tab"

    def get_policy(self, x: int) -> npt.ArrayLike:
        probs = np.zeros(self.num_actions)
        probs += self.epsilon / (self.num_actions)
        a = np.argmax(self.behaviour_learner.get_action_values(x))
        probs[a] += 1 - self.epsilon
        return probs

    # public method for rlglue
    def selectAction(self, x: int) -> Tuple[int, int] :
        a = self.random.choice(self.num_actions, p = self.get_policy(x))
        return a

    def update(self, x, a, xp, r, gamma):
        # not strictly needed because the option action pair shouldn't be used in termination,
        # but it prevents some unneeded computation that could error out with weird indexing.
        action = self.selectAction(xp) if xp != options.GRAZING_WORLD_TERMINAL_STATE else None

        self.tau += 1
        self.tau[x, a] = 0

        # Direct RL
        self.behaviour_learner.update(x, a, xp, r, gamma, self.alpha)

        self.update_model(x,a,xp,r)  
        self.planning_step(x, xp)

        return action

    def update_model(self, x, a, xp, r):
        """updates the model 
        
        Returns:
            Nothing
        """

        self.action_model.update(x, a, xp, r)
        if isinstance(self.option_model, OptionModel_Sutton_Tabular):
            self.option_model.update(x, a, xp, r, self.gamma, self.alpha)
        else:
            raise NotImplementedError(f'Update for {type(self.option_model)} is not implemented')
        
    def option_model_planning_update(self, x, a, xp ,r):
        if isinstance(self.option_model, OptionModel_Sutton_Tabular):
            self.option_model.update(x, a, xp, r, self.gamma, self.alpha)
        else:
            raise NotImplementedError(f'Planning update for {type(self.option_model)} is not implemented')

    def planning_step(self, x: int, xp: int):
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

        for _ in range(self.planning_steps):
            if self.search_control=="random":
                plan_x = choice(np.array(self.action_model.visited_states()), self.random)
            elif self.search_control =="current":
                visited_states = list(self.action_model.visited_states())
                if (xp in list(visited_states)):
                    plan_x = xp
                else:
                    # Random if you haven't visited the next state yet
                    plan_x = x
            
            # Random action
            a = self.random.choice(self.action_model.visited_actions(plan_x))
            xp, r = self.action_model.predict(plan_x, a)

            xp_values = []
            xp_values.append(np.max(self.behaviour_learner.get_action_values(xp)))

            for o in range(self.num_options):
                # Generating the experience from the option model
                r_option, discount, transition_prob = self.option_model.predict(xp, o)
                transition_prob = np.clip(transition_prob, a_min = 0, a_max = None)
                norm = np.linalg.norm(transition_prob, ord=1)
                if (norm != 0):
                    prob = transition_prob / norm
                    # +1 here accounts for the terminal state
                    xpp = self.random.choice(self.num_states + 1, p=prob)
                    xpp_prediction = np.max(self.behaviour_learner.get_action_values(xpp))
                    xp_values.append(r_option + discount * xpp_prediction)
            
            # Exploration bonus for +
            r += self.kappa * np.sqrt(self.tau[x, o])

            target = r + self.gamma * np.max(xp_values)

            if isinstance(self.behaviour_learner, QLearner):
                self.behaviour_learner.target_update(plan_x, a, target, self.alpha)
            elif isinstance(self.behaviour_learner, ESarsaLambda):
                self.behaviour_learner.planning_update(x, o, xp, r, self.get_policy(xp), discount, self.alpha)
            else:
                raise NotImplementedError()
        


    def agent_end(self, x, a, r, gamma):
        self.update(x, a, options.GRAZING_WORLD_TERMINAL_STATE, r, gamma)