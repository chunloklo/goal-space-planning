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
from src.agents.components.models import OptionModel_Sutton_Tabular, CombinedModel_ESarsa_Tabular
from src.agents.components.approximators import DictModel

class Dyna_Tab:
    def __init__(self, features: int, actions: int, params: Dict, seed: int, options, env):
        self.wrapper_class = rlglue.OneStepWrapper

        self.env = env
        self.features = features
        self.num_actions = actions
        self.actions = list(range(self.num_actions))
        self.params = params
        self.num_states = self.env.nS
        self.random = np.random.RandomState(seed)

        # This is only needed for RL glue (for convenience reason to have uniform initialization between
        # options and non-options agents). Not used in the algorithm at all.
        self.options = options

        # define parameter contract
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']
        self.planning_steps = params['planning_steps']
        self.gamma = params['gamma']
        self.kappa = params['kappa']
        self.lmbda = params['lambda']
        self.behaviour_alg = param_utils.parse_param(params, 'behaviour_alg', lambda p : p in ['QLearner', 'ESarsaLambda']) 
        self.search_control = param_utils.parse_param(params, 'search_control', lambda p : p in ['random', 'current', 'close'])
        
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

        # For 'close' search control
        self.distance_from_goal = {}

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
    def selectAction(self, x: int) -> Tuple[int, int] :
        a = self.random.choice(self.num_actions, p = self.get_policy(x))
        return a

    def update(self, x, a, xp, r, gamma):

        if (xp != options.GRAZING_WORLD_TERMINAL_STATE):
            ap = self.selectAction(xp)
        else:
            ap = None

        # Exploration bonus tracking
        self.tau += 1
        self.tau[x, a] = 0

        if isinstance(self.behaviour_learner, QLearner):
            self.behaviour_learner.update(x, a, xp, r, gamma, self.alpha)
        elif isinstance(self.behaviour_learner, ESarsaLambda):
            self.behaviour_learner.update(x, a, xp, r, self.get_policy(xp), self.gamma, self.lmbda, self.alpha)
        else:
            raise NotImplementedError()

        self.update_model(x,a,xp,r)  
        self.planning_step(x, xp, self.search_control)

        return ap
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

    def _planning_update(self, x: int, a: int):
        if x == options.GRAZING_WORLD_TERMINAL_STATE:
            # If its terminal state, no need to plan with it
            return

        xp, r = self.action_model.predict(x, a)
        discount = self.gamma

        # Exploration bonus for +
        r += self.kappa * np.sqrt(self.tau[x, a])

        if isinstance(self.behaviour_learner, QLearner):
            self.behaviour_learner.planning_update(x, a, xp, r, discount, self.alpha)
        elif isinstance(self.behaviour_learner, ESarsaLambda):
            self.behaviour_learner.planning_update(x, a, xp, r, self.get_policy(xp), discount, self.alpha)
        else:
            raise NotImplementedError()

    def planning_step(self, x:int, xp: int, search_control: str):
        """performs planning, i.e. indirect RL.

        Returns:
            Nothing
        """

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
            
            # Pick a random action/option within all eligable action/options
            # I think there should be a better way of doing this...
            visited_actions = self.action_model.visited_actions(plan_x)

            for a in visited_actions: 
                self._planning_update(plan_x, a)

    def agent_end(self, x, a, r, gamma):
        self.update(x, a, options.GRAZING_WORLD_TERMINAL_STATE, r, gamma)
        self.behaviour_learner.episode_end()