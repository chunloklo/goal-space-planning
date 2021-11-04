from typing import Any, List, Dict, Tuple
import numpy as np
import numpy.typing as npt

import random
from utils import numpy_utils, param_utils
from src.agents.components.traces import Trace
from src.agents.components.approximators import TabularApproximator
from src.utils.feature_utils import stacked_tabular_features, stacked_features
from src.utils import globals
from src.utils import options
from utils.Option import Option
import heapq

class QLearner():
    def __init__(self, num_state_features: int, num_actions: int, pq_size:int):
        self.num_state_features: int = num_state_features
        self.num_actions: int = num_actions
        self.Q = np.zeros((self.num_state_features, self.num_actions))
        # self.update_counter = 0
        # self.average_delta = 0
        # self.stop_counter = 0

        self.priority_q_size = pq_size
        self.priority_q = []
        for i in range(self.priority_q_size):
            self.priority_q.append([0,90]) # 90 for start state, this should be changed later for an (inherited) variable, 0 is for TD error

    def get_action_values(self, x: int) -> np.ndarray:
        return self.Q[x, :]
    
    def get_priority_q(self):
        return [i[1] for i in self.priority_q]

    def replace_element_in_pq(self,delta, xp):
        for i,element in enumerate(self.priority_q):
            if xp == element[1]:
                self.priority_q[i][0] = delta

    def update_priority_q(self,delta,xp):
        if delta>self.priority_q[0][0] and xp not in self.get_priority_q(): 
            self.priority_q.pop(0)
            self.priority_q.append([delta, xp])
        elif xp in self.get_priority_q():
            self.replace_element_in_pq(delta,xp)
        self.priority_q.sort()
        
    def planning_update(self, x: int, a: int, xp: int, r: float, env_gamma: float, step_size: float):
        self.update(x, a, xp, r, env_gamma, step_size)
    
    def update(self, x: int, a: int, xp: int, r: float, env_gamma: float, step_size: float):
        x_prediction = self.Q[x, a]
        xp_predictions = self.get_action_values(xp)

        if random.random() > 0.8:
            max_q  = np.average(xp_predictions)
        else:
            max_q = np.max(xp_predictions)
        
        delta = r + env_gamma * max_q - x_prediction
        # 100 for terminal state, again, this should be a variable
        if xp !=100:
            self.update_priority_q(abs(delta),xp)
        # self.average_delta = (self.average_delta * self.update_counter + delta) / (self.update_counter +1)
        # self.update_counter+=1
        # self.stop_counter+=1
        # if abs(delta) > self.average_delta and self.stop_counter>=100:
        #     self.Q[:,4:7] += delta
        #     self.stop_counter=0
        

        self.Q[x, a] += step_size * delta

    def episode_end(self):
        globals.collector.collect('Q', np.copy(self.Q))   
        pass

class ESarsaLambda():
    def __init__(self, num_state_features: int, num_actions: int):
        self.num_state_features: int = num_state_features
        self.num_actions: int = num_actions

        # Only thing we want to estimate is the value
        self.trace = Trace(self.num_state_features * self.num_actions, 1, 'tabular')

        # Could change based on what option model we want.
        self.approximator = TabularApproximator(self.num_state_features * self.num_actions, 1)

    def get_action_values(self, x: int) -> np.ndarray:
        xa_indices = []
        for a in range(self.num_actions):
            xpap_index = stacked_tabular_features(x, a, self.num_state_features)
            xa_indices.append(xpap_index)
        
        return self.approximator.predict(xa_indices)

    def planning_update(self, x: int, a: int, xp: int, r: float, xp_policy: npt.ArrayLike, env_gamma: float, step_size: float):
        xa_index = stacked_tabular_features(x, a, self.num_state_features)
        xp_predictions = self.get_action_values(xp)

        # ESarsa target based on policy
        xp_average = np.average(xp_predictions, weights=xp_policy)

        # Constructing the delta vector
        target = r + env_gamma * xp_average

        grad = self.approximator.grad(xa_index, target)
        self.approximator.update(step_size, grad)

    def update(self, x: int, a: int, xp: int, r: float, xp_policy: npt.ArrayLike, env_gamma: float, lmbda: float, step_size: float):
        xa_index = stacked_tabular_features(x, a, self.num_state_features)
        x_predictions = self.approximator.predict(xa_index)
        xp_predictions = self.get_action_values(xp)

        # ESarsa target based on policy
        xp_average = np.average(xp_predictions, weights=xp_policy)

        # Constructing the delta vector
        delta = r + env_gamma * xp_average - x_predictions


        self.trace.update(env_gamma, lmbda, xa_index, 1)
        self.approximator.update(step_size, delta * self.trace.z)

    def episode_end(self):
        weights = self.approximator.weights
        # print(weights.shape)
        # This reshape order is specific to how the features are stacked (action follow each other)
        q = weights.reshape((self.num_state_features, self.num_actions), order='F')
        globals.collector.collect('Q', np.copy(q))   
        self.trace.episode_end()