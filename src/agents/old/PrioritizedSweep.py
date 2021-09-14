from typing import Dict
import numpy as np
from PyExpUtils.utils.random import argmax, choice
import random
from src.utils.PriorityQueue import PriorityQueue

class PrioritizedSweep:
    def __init__(self, features: int, actions: int, params: Dict, seed: int):
        self.features = features
        self.actions = actions
        self.params = params
        self.PQ  = PriorityQueue()
        self.random = np.random.RandomState(seed)

        # define parameter contract
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']
        self.planning_steps = params['planning_steps']
        self.size = params['size']

        self.gamma = params['gamma']
        self.theta = 0.2 # threshold value for inserting into the PriorityQueue
        self.a = -1
        self.x = -1

        self.Q = np.zeros((int(features/actions), actions))
        self.model = {}
        self.inverse_model = {}

    def FA(self):
        return "Tabular"

    def __str__(self):
        return "PrioritizedSweep"

    def selectAction(self, x):
        p = self.random.rand()
        if p < self.epsilon:
            return choice(np.arange(self.actions), self.random)
        return argmax(self.Q[x,:])

    def update(self, x, a, xp, r, gamma):

        ap = self.selectAction(xp)
        self.update_model(x,a,xp,r) 
        self.update_inverse(x,a,xp)
        P =  abs(r + gamma*np.max(self.Q[xp,:]) - self.Q[x,a])
        if P > self.theta:
            self.PQ.insert((x,a,P))


        self.planning_step(gamma)

        #self.Q[x, a] = self.Q[x,a] + self.alpha * (r + gamma*np.max(self.Q[xp,:]) - self.Q[x,a])   
        return ap

    def update_model(self, x, a, xp, r):
        """updates the model 
        
        Returns:
            Nothing
        """
        if x not in self.model:
            self.model[x] = {}
        self.model[x][a] = (xp,r)

    def update_inverse(self,x,a,xp):
        """
        Returns the states and actions that would lead to xp according to the model
        """
        if xp not in self.inverse_model:
            self.inverse_model[xp] = set()
        self.inverse_model[xp].add((x,a))
        # try:
        #     print(self.inverse_model[-1])
        # except:
        #     pass

    def planning_step(self,gamma):
        """performs planning, i.e. indirect RL.
        Returns:
            Nothing
        """
        
        for _ in range(self.planning_steps):
            if self.PQ.isEmpty():
                break
            
            x, a, _ = self.PQ.delete()
            xp, r = self.model[x][a]
            if xp == -1:
                max_q = 0
            else:
                max_q = np.max(self.Q[xp,:])
            #print(self.inverse_model)

            self.Q[x,a] = self.Q[x,a] + self.alpha * (r + gamma * max_q - self.Q[x, a])

            if xp != -1:
                for S_Bar, A_Bar in self.inverse_model[xp]:
                    _, R_Bar = self.model[x][a]
                    P = abs(R_Bar + gamma * np.max(self.Q[xp,:]) - self.Q[S_Bar, A_Bar])

                    if P > self.theta:
                        self.PQ.insert((S_Bar,A_Bar,P))

    def agent_end(self, x, a, r, gamma):
        # Model Update step
        self.update(x,a,-1,r,gamma)
        #self.update_model(x,a, -1, r)
        # Planning
        #self.planning_step(gamma)