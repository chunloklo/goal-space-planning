from typing import Dict
import numpy as np
from PyExpUtils.utils.random import argmax, choice

class Q_Tabular:
    def __init__(self, features: int, actions: int, params: Dict, seed: int, options, env):
        self.features = features
        self.num_actions = actions
        self.actions = list(range(self.num_actions))
        self.params = params
        self.random = np.random.RandomState(seed)
        self.options = options

        # define parameter contract
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']

        # create initial weights
        self.Q = np.zeros((int(features/actions), actions))

    def FA(self):
        return "Tabular"

    def __str__(self):
        return "Q_Tabular"

    def selectAction(self, x):
        p = self.random.rand()
        if p < self.epsilon:
            return choice(np.arange(self.num_actions), self.random)
        return argmax(self.Q[x,:])

    def update(self, x, a, xp, r, gamma):
        ap = self.selectAction(xp)
        self.Q[x, a] = self.Q[x,a] + self.alpha * (r + gamma*np.max(self.Q[xp,:]) - self.Q[x,a])   
        return ap