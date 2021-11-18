from typing import Dict, Union, Tuple
import numpy as np
import numpy.typing as npt
from PyExpUtils.utils.random import argmax, choice
import random
from PyFixedReps.Tabular import Tabular
from agents.components.learners import ESarsaLambda, QLearner
from agents.components.search_control import ActionModelSearchControl_Tabular
from src.utils import rlglue, feature_utils
from src.utils import globals
from src.utils import options, param_utils
from src.agents.components.models import OptionModel_Sutton_Tabular, CombinedModel_ESarsa_Tabular
from src.agents.components.approximators import TabularApproximator
from utils import numpy_utils
from utils.numpy_utils import create_onehot

class ActorCritic_Tab:
    def __init__(self, features: int, actions: int, params: Dict, seed: int, options, env):
        self.wrapper_class = rlglue.OneStepWrapper

        self.options = None

        self.env = env
        self.features = features
        self.num_actions = actions
        self.actions = list(range(self.num_actions))
        self.params = params
        self.num_states = self.env.nS + 1
        self.random = np.random.RandomState(seed)



        # define parameter contract
        self.alpha = params['alpha']
        self.gamma = params['gamma']

        self.policy_approximator = TabularApproximator(self.num_states * self.num_actions, 1)
        self.value_estimator = TabularApproximator(self.num_states, 1)

    def FA(self):
        return "Tabular"

    def __str__(self):
        return "ActorCritic_Tab"

    # public method for rlglue
    def selectAction(self, x: int) -> Tuple[int, int] :
        # a = self.random.choice(self.num_actions, p = self.get_policy(x))
        b_preferences = []
        for b in range(self.num_actions):
            preference = self.policy_approximator.predict(feature_utils.stacked_tabular_features(x, b, self.num_states))
            b_preferences.append(preference)

        prob = numpy_utils.softmax(np.array(b_preferences))
        a = self.random.choice(self.num_actions, p=prob.flatten())
        return a

    def update(self, x, a, xp, r, gamma):
        delta = r + gamma * self.value_estimator.predict(xp) - self.value_estimator.predict(x)
        value_grad = self.value_estimator.grad(create_onehot(self.num_states, x))
        self.value_estimator.update(self.alpha, delta * value_grad)

        
        b_features = []
        b_preferences = []
        for b in range(self.num_actions):
            feature = feature_utils.stacked_tabular_features(x, b, self.num_states)
            b_features.append(feature)
            preference = self.policy_approximator.predict(feature)
            b_preferences.append(preference)
        
        b_preferences = np.array(b_preferences)
        b_policy = numpy_utils.softmax(b_preferences)
        
        policy_grad = numpy_utils.create_onehot(self.num_states * self.num_actions, feature_utils.stacked_tabular_features(x, a, self.num_states))
        # print(policy_grad)
        for b in range(self.num_actions):
            policy_grad -= b_policy[b] * numpy_utils.create_onehot(self.num_states * self.num_actions, b_features[b])


        policy_grad = np.expand_dims(policy_grad, 1)
        # print(policy_grad.shape)
        # print(self.policy_approximator.weights.shape)

        self.policy_approximator.update(self.alpha, delta * policy_grad)




        ap = self.selectAction(xp)
        return ap
    
    def agent_end(self, x, a, r, gamma):
        self.update(x, a, globals.blackboard['terminal_state'], r, gamma)

        action_preferences = np.zeros((self.num_states, self.num_actions))
        for s in range(self.num_states):
            for a in range(self.num_actions):
                # print(';ll')
                # print(self.num_states * self.num_actions)
                # print(feature_utils.stacked_tabular_features(s, a, self.num_states))
                action_preferences[s, a] = self.policy_approximator.predict(feature_utils.stacked_tabular_features(s, a, self.num_states))
        globals.collector.collect('action_preferences', np.copy(action_preferences))   