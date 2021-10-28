from typing import Dict
import numpy as np
from PyExpUtils.utils.random import argmax, choice
import random
from src.utils import rlglue, param_utils
from src.agents.components.approximators import DictModel

class Dynaqp_Tab:
    def __init__(self, features: int, actions: int, params: Dict, seed: int, options, env):
        self.wrapper_class = rlglue.OneStepWrapper
        self.features = features
        self.num_actions = actions
        self.params = params
        self.actions = list(range(actions))
        self.random = np.random.RandomState(seed)

        self.options = options

        # define parameter contract
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']
        self.planning_steps = params['planning_steps']
        # Whether to plan with current state.
        self.plan_with_current_state = param_utils.parse_param(params, 'planning_method', lambda p : p in ['random', 'current'])


        self.gamma = params['gamma']
        self.kappa = params['kappa']
        self.tau = np.zeros(((int(features/self.num_actions), self.num_actions)))
        self.a = -1
        self.x = -1

        self.Q = np.zeros((int(features/self.num_actions), self.num_actions))

        self.model = DictModel()
    def FA(self):
        return "Tabular"

    def __str__(self):
        return "Dynaqp_Tab"

    def selectAction(self, x):
        p = self.random.rand()
        if p < self.epsilon:
            return choice(np.arange(self.num_actions), self.random)
        return argmax(self.Q[x,:])

    def update(self, x, a, xp, r, gamma):
        ap = self.selectAction(xp)
        self.tau += 1
        self.tau[x, a] = 0
        max_q = 0 if xp == -1 else np.max(self.Q[xp,:])
        self.Q[x, a] = self.Q[x,a] + self.alpha * (r + gamma*max_q - self.Q[x,a]) 
        self.update_model(x,a,xp,r)  
        if self.plan_with_current_state == 'current':
            self.planning_with_current_state(x)
        elif self.plan_with_current_state == 'random':
            self.planning_step()
        return ap

    def update_model(self, x, a, xp, r):
        """updates the model 
        
        Returns:
            Nothing
        """

        # Keeping this as a separate function so we could add more things if we need later on
        self.model.update(x, a, xp, r)

    def planning_with_current_state(self, x):
        for i in range(self.planning_steps):
            a = choice(self.model.visited_actions(x), self.random)
            xp, r = self.model.predict(x, a)
            r+= self.kappa * np.sqrt(self.tau[x, a])

            if xp ==-1:
                max_q = 0
            else:
                max_q = np.max(self.Q[xp,:])
            
            self.Q[x,a] = self.Q[x,a] + self.alpha * (r + self.gamma * max_q - self.Q[x, a])
            
            # if self.Q[x,a]>50:
            #     print(x,a)
            #     print(self.Q[x,a])


    def planning_step(self):
        """performs planning, i.e. indirect RL.

        Returns:
            Nothing
        """

        for _ in range(self.planning_steps):
            x = choice(self.model.visited_states(), self.random)
            a = choice(self.model.visited_actions(x), self.random) 
            xp, r = self.model.predict(x, a)
            r+= self.kappa * np.sqrt(self.tau[x, a])

            if xp ==-1:
                max_q = 0
            else:
                max_q = np.max(self.Q[xp,:])
            
            self.Q[x,a] = self.Q[x,a] + self.alpha * (r + self.gamma * max_q - self.Q[x, a])
        
    def agent_end(self, x, a, r, gamma):
        self.update(x, a, -1, r, gamma)

