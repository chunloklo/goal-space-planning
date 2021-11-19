from typing import Any, List, Dict, Tuple
import numpy as np
import numpy.typing as npt

from utils import numpy_utils, param_utils
from src.agents.components.traces import Trace
from src.agents.components.approximators import TabularApproximator
from src.utils.feature_utils import stacked_tabular_features, stacked_features
from src.utils import globals
from src.utils import options
from utils.Option import Option
import haiku as hk
import jax.numpy as jnp
import jax
import optax

class QLearner_ImageNN_funcs():
    def __init__(self, num_actions: int):
        self.num_actions = num_actions

        def q_function(states):
            mlp = hk.Sequential([
                hk.Linear(40), jax.nn.relu,
                hk.Linear(20), jax.nn.relu,
                hk.Linear(self.num_actions),
            ])
            return mlp(states) 
        self.f_qfunc = hk.without_apply_rng(hk.transform(q_function))
        self.f_opt = optax.adam(1e-3)

        def get_q_values(params: hk.Params, x: Any):
            action_values = self.f_qfunc.apply(params, x)
            return action_values

        def loss(params: hk.Params, x, a, xp, r, gamma):
            x = x.flatten()
            if xp is not None:
                xp = xp.flatten()
                pred = self.f_qfunc.apply(params, x)
                xp_pred = jnp.max(jax.lax.stop_gradient(self.f_qfunc.apply(params, xp)))
                return  jnp.square(r + gamma * xp_pred - pred[a])
            else:
                pred = self.f_qfunc.apply(params, x)
                return  jnp.square(r - pred[a])

        def update(params: hk.Params, opt_state, x, a, xp, r, gamma):
            grads = jax.grad(loss)(params, x, a, xp, r, gamma)
            updates, opt_state = self.f_opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state

        self.f_get_q_values = jax.jit(get_q_values)
        self.f_update = jax.jit(update)        
        return

class QLearner_ImageNN():
    def __init__(self, image_size: Tuple[int, int], num_actions: int):
        self.image_size: int = image_size
        self.num_actions: int = num_actions
        # For now, flatten the image
        self.input_size = image_size[0] * image_size[1]

        self.funcs = QLearner_ImageNN_funcs(num_actions)

        dummy_state = jnp.zeros(self.input_size)
        self.params = self.funcs.f_qfunc.init(jax.random.PRNGKey(42), dummy_state)
        self.opt_state = self.funcs.f_opt.init(self.params)
    
    def get_action_values(self, x: npt.ArrayLike) -> np.ndarray:
        action_values = self.funcs.f_get_q_values(self.params, x.flatten())
        return action_values

    def update(self, x: int, a: int, xp: int, r: float, env_gamma: float, step_size: float):
        self.params, self.opt_state = self.funcs.f_update(self.params, self.opt_state, x, a, xp, r, env_gamma)
        return self.params

class QLearner():
    def __init__(self, num_state_features: int, num_actions: int):
        self.num_state_features: int = num_state_features
        self.num_actions: int = num_actions
        self.Q = np.zeros((self.num_state_features, self.num_actions))

    def get_action_values(self, x: int) -> np.ndarray:
        return self.Q[x, :]
    
    def planning_update(self, x: int, a: int, xp: int, r: float, env_gamma: float, step_size: float):
        self.update(x, a, xp, r, env_gamma, step_size)

    def target_update(self, x: int, a: int, target: float, step_size: float):
        x_prediction = self.Q[x, a]
        delta = target - x_prediction
        self.Q[x, a] += step_size * delta

    def update(self, x: int, a: int, xp: int, r: float, env_gamma: float, step_size: float):
        x_prediction = self.Q[x, a]
        xp_predictions = self.get_action_values(xp)

        max_q = np.max(xp_predictions)
        delta = r + env_gamma * max_q - x_prediction
        self.Q[x, a] += step_size * delta

        # Pushing up learner delta for search control
        globals.blackboard['learner_delta'] = delta

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

    def update(self, x: int, a: int, xp: int, r: float, xp_policy: npt.ArrayLike, env_gamma: float, lmbda: float, step_size: float, terminal: bool):
        xa_index = stacked_tabular_features(x, a, self.num_state_features)
        x_predictions = self.approximator.predict(xa_index)
        xp_predictions = self.get_action_values(xp)

        # ESarsa target based on policy
        xp_average = np.average(xp_predictions, weights=xp_policy)

        # Constructing the delta vector
        delta = r + env_gamma * xp_average - x_predictions

        self.trace.update(env_gamma, lmbda, xa_index, 1)
        self.approximator.update(step_size, delta * self.trace.z)

        # Pushing up learner delta for search control
        globals.blackboard['learner_delta'] = delta

    def episode_end(self):
        weights = self.approximator.weights
        # print(weights.shape)
        # This reshape order is specific to how the features are stacked (action follow each other)
        q = weights.reshape((self.num_state_features, self.num_actions), order='F')
        globals.collector.collect('Q', np.copy(q))   
        self.trace.episode_end()