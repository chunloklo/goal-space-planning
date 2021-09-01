from typing import Dict
import numpy as np
from PyExpUtils.utils.random import argmax, choice
import random


class Dynaqp_Tab:
    def __init__(self, features: int, actions: int, params: Dict, seed: int, options, env):
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



        self.gamma = params['gamma']
        self.kappa = params['kappa']
        self.tau = np.zeros(((int(features/self.num_actions), self.num_actions)))
        self.a = -1
        self.x = -1

        self.Q = np.zeros((int(features/self.num_actions), self.num_actions))
        self.model = {}
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
        self.Q[x, a] = self.Q[x,a] + self.alpha * (r + gamma*np.max(self.Q[xp,:]) - self.Q[x,a])   
        self.update_model(x,a,xp,r)  
        self.planning_step(gamma)
        return ap

    def update_model(self, x, a, xp, r):
        """updates the model 
        
        Returns:
            Nothing
        """
        
        if x not in self.model:
            self.model[x] = {a:(xp,r)}
            for action in self.actions:
                if action != a:
                    self.model[x][action] = (x, 0)
        else:
            self.model[x][a] = (xp,r)
            

    def planning_step(self,gamma):
        """performs planning, i.e. indirect RL.

        Returns:
            Nothing
        """

        for i in range(self.planning_steps):
            x = choice(np.array(list(self.model.keys())), self.random)
            a = choice(np.array(list(self.model[x].keys())), self.random) 
            xp, r = self.model[x][a]
            r+= self.kappa * np.sqrt(self.tau[x, a])

            if xp ==-1:
                max_q = 0
            else:
                max_q = np.max(self.Q[xp,:])
            
            self.Q[x,a] = self.Q[x,a] + self.alpha * (r + gamma * max_q - self.Q[x, a])
            
            # if self.Q[x,a]>50:
            #     print(x,a)
            #     print(self.Q[x,a])
            

    def agent_end(self, x, a, r, gamma):
        # Model Update step
        self.update_model(x,a, -1, r)
        # Planning
        self.planning_step(gamma)
