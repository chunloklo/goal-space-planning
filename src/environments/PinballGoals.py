"""
.. module:: pinball
   :platform: Unix, Windows
   :synopsis: Pinball domain for reinforcement learning

.. moduleauthor:: Pierre-Luc Bacon <pierrelucbacon@gmail.com>


"""

import random
import argparse, os
from re import S
import numpy as np
from RlGlue import BaseEnvironment
from itertools import *

class PinballGoals():

    termination_radius = 0.04
    initiation_radius = 0.45
    speed_radius = 0.2
    
    # Caching calculation for functions
    termination_radius_squared = np.square(termination_radius)
    initiation_radius_squared = np.square(initiation_radius)
    speed_radius_squared = np.square(speed_radius)

    def __init__(self):
        self.goals = []
        # n is the number of goals "per side". Total number of goals is n x n
        n = 4
        border = 0.06

        for y in np.linspace(0 + border, 1 - border, n):
            for x in np.linspace(0 + border, 1 - border, n):
                self.goals.append([x,y])

        self.goals = np.array(self.goals)
        self.num_goals = self.goals.shape[0]

        # Shifting away some goals so they aren't on top of obstacles
        self.goals[9] += [-0.06, 0.0]
        self.goals[10] += [-0.08, 0.0]
        self.goals[11] += [0.0, 0.06]

        self.goal_speeds = np.zeros(self.goals.shape)
        
    def goal_termination(self, s):
        state_close = np.sum(np.power(self.goals - s[:2], 2), axis=1) <= self.termination_radius_squared
        speed_close = np.sum(np.power(s[2:] - self.goal_speeds, 2), axis=1) <= self.speed_radius_squared
        terms = np.logical_and(state_close, speed_close)
        return terms

    def goal_initiation(self, s):
        state_close = np.sum(np.power(self.goals - s[:2], 2), axis=1) <= self.initiation_radius_squared
        return state_close

class PinballOracleGoals(PinballGoals):
    def __init__(self):
        super().__init__()
        self.goals
        oracle_adjusted_goals = {
            9: np.array([0.26634864519615703, 0.6574340050572484, 0.38341283870096077, -0.9752487531218751]),
            5: np.array([0.35442983272352163, 0.34282851754474797, 0.7613925418191196, -0.7762487531218751]),
            6: np.array([0.6409535893184335, 0.3271380768559643, 0.995, -0.17232614294967943])
        }

        for goal_num in oracle_adjusted_goals:
            self.goals[goal_num] = oracle_adjusted_goals[goal_num][:2]
            self.goal_speeds[goal_num] = oracle_adjusted_goals[goal_num][2:]
