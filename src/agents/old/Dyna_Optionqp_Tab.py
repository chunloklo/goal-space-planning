from typing import Dict
import numpy as np
from PyExpUtils.utils.random import argmax, choice
import random


class Dyna_Optionqp_Tab:
    def __init__(self, features: int, actions: int, params: Dict, seed: int, options, env):
        self.env = env
        self.features = features
        self.num_actions = actions
        self.actions = list(range(self.num_actions))
        self.params = params
        self.num_states = self.env.nS
        self.options = options
        self.num_options = len(self.env.terminal_state_positions)
        self.random = np.random.RandomState(seed)

        # define parameter contract
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']
        self.planning_steps = params['planning_steps']
        self.gamma = params['gamma']
        self.kappa = params['kappa']

        self.tau = np.zeros(((int(features/self.num_actions), self.num_actions)))
        self.a = -1
        self.x = -1

        # create initial weights
        self.Q = np.zeros((self.num_states, self.num_actions+self.num_options))
        # initiation set is 1, all options are pertinent everywhere
        self.d = np.ones((self.num_states, self.num_options))
        # termination condition is the environment's discount except in terminal states where it's 0
        self.term_cond = np.ones((self.num_states, self.num_options))*params["gamma"]
        for terminal_state_position in self.env.terminal_state_positions:
            self.Q[self.env.state_encoding(terminal_state_position),:]=0

        self.option_Qs = [np.zeros((self.num_states, self.num_actions)) for _ in range(self.num_options)]

        self.model = {}
    def FA(self):
        return "Tabular"

    def __str__(self):
        return "Dyna_Optionqp_Tab"

    def selectAction(self, x):
        p = self.random.rand()
        if p < self.epsilon:
            o = choice(np.arange(self.num_actions+self.num_options), self.random)
        else:
            o = argmax(self.Q[x,:])

        if o>= self.num_actions:
            a = argmax(self.option_Qs[(self.num_actions+self.num_options)-o-1][x,:])
        else:
            a=o
        return o,a

    def update(self, x,o, a, xp, r, gamma):
        op, ap = self.selectAction(xp)
        self.tau += 1
        self.tau[x, a] = 0
        self.Q[x, a] = self.Q[x,a] + self.alpha * (r + gamma*np.max(self.Q[xp,:]) - self.Q[x,a]) 
        if o!=None and o >= self.num_actions:
            Q_opt = self.option_Qs[(self.num_actions+self.num_options)-o-1]
            Q_opt[x,a] += self.alpha * (r + gamma*np.max(Q_opt[xp,:]) - Q_opt[x,a]) 
        self.update_model(x,a,xp,r)  
        self.planning_step(gamma)

        return op, ap

    def update_model(self, x, o, xp, r):
        """updates the model 
        
        Returns:
            Nothing
        """
        
        if x not in self.model:
            self.model[x] = {o:(xp,r)}
            for action in self.actions:
                if action != o:
                    self.model[x][action] = (x, 0)
        else:
            self.model[x][o] = (xp,r)
            

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
            #     print(self.Q[x,a])
            #     print(x,a)
            

    def agent_end(self, x, a, r, gamma):
        # Model Update step
        self.update_model(x,a, -1, r)
        # Planning
        self.planning_step(gamma)