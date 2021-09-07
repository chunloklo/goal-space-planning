import numpy as np
from typing import Optional, Sequence
from numba import njit
from PyFixedReps.BaseRepresentation import BaseRepresentation

class Tabular(BaseRepresentation):
    def __init__(self, shape, actions):
        self.shape = shape
        self.actions = actions


    def features(self):
        return int(self.shape[0] * self.shape[1] * self.actions)

    def encode(self, s, a = None):
        return s[0]*self.shape[0]+s[1]


