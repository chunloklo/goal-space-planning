from typing import Any, List, Dict, Literal, Tuple
import numpy as np
import numpy.typing as npt
from optax._src.update import incremental_update

from src.utils import numpy_utils, param_utils
from src.agents.components.traces import Trace
from src.agents.components.approximators import TabularApproximator
from src.utils.feature_utils import stacked_tabular_features, stacked_features
from src.utils import globals
from src.utils import options
from src.utils.Option import Option
import haiku as hk
import jax.numpy as jnp
import jax
import optax
import copy

class QLearner_ImageNN_funcs():
    def __init__(self, num_actions: int, learning_rate: float):
        self.num_actions = num_actions

        def q_function(states):
            mlp = hk.Sequential([
                hk.Conv2D(32, [3, 3], stride=1), jax.nn.relu,
                hk.Flatten(),
                hk.Linear(64), jax.nn.relu,
                hk.Linear(32), jax.nn.relu,
                hk.Linear(self.num_actions),
            ])
            return mlp(states) 
        self.f_qfunc = hk.without_apply_rng(hk.transform(q_function))
        self.f_opt = optax.adam(learning_rate)

        def get_q_values(params: hk.Params, x: Any):
            action_values = self.f_qfunc.apply(params, x)
            return action_values
        
        def get_td_errors(params: hk.Params, target_params: hk.Params, data):
            r = data['r']
            x = data['x']
            a = data['a']
            xp = data['xp']
            gamma = data['gamma']

            x_pred = self.f_qfunc.apply(params, x)
            xp_pred = jax.lax.stop_gradient(jnp.max(self.f_qfunc.apply(target_params, xp), axis=1))
            prev_pred = jnp.take_along_axis(x_pred, jnp.expand_dims(a, axis=1), axis=1).squeeze()
            delta = r + gamma * xp_pred - prev_pred
            return delta
        
        def loss_option_target(params: hk.Params, target_params: hk.Params, data):
            # There is some copypasta here which isn't great. 
            r = data['r']
            x = data['x']
            a = data['a']
            xp = data['xp']
            gamma = data['gamma']
            option_q_sa = data['best_option_q_sa']

            x_pred = self.f_qfunc.apply(params, x)
            xp_pred = jax.lax.stop_gradient(jnp.max(self.f_qfunc.apply(target_params, xp), axis=1))
            prev_pred = jnp.take_along_axis(x_pred, jnp.expand_dims(a, axis=1), axis=1).squeeze()
            td_errors = jnp.maximum(r + gamma * xp_pred, option_q_sa) - prev_pred
            return jnp.mean(jnp.square(td_errors)), td_errors

        def update_option_target(params: hk.Params, target_params: hk.Params, opt_state, data):
            grads, td_errors = jax.grad(loss_option_target, has_aux=True)(params, target_params, data)
            updates, opt_state = self.f_opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, td_errors

        def loss(params: hk.Params, target_params: hk.Params, data):
            td_errors = get_td_errors(params, target_params, data)
            return  jnp.mean(jnp.square(td_errors)), td_errors

        def update(params: hk.Params, target_params: hk.Params, opt_state, data):
            grads, td_errors = jax.grad(loss, has_aux=True)(params, target_params, data)
            updates, opt_state = self.f_opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, td_errors


        # For the shift update
        def shift_loss(params: hk.Params, data):
            # There is some copypasta here which isn't great. 
            x = data['x']
            best_option_values = data['best_option_values']

            x_pred = self.f_qfunc.apply(params, x)
            return jnp.mean(jnp.square(x_pred - jnp.maximum(jax.lax.stop_gradient(x_pred), best_option_values)))
        
        def shift_update(params: hk.Params, opt_state, data):
            grads = jax.grad(shift_loss)(params, data)
            updates, opt_state = self.f_opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state

        self.f_get_q_values = jax.jit(get_q_values)
        self.f_update = jax.jit(update)  
        self.f_update_target = jax.jit(update_option_target)
        self.f_get_td_errors = jax.jit(get_td_errors)
        self.f_shift_update = jax.jit(shift_update)      
        return

class QLearner_ImageNN():
    def __init__(self, image_size: Tuple[int, int], num_actions: int, learning_rate: float):
        # One color channel for B&W
        self.image_size: int = image_size
        self.num_actions: int = num_actions
        self.funcs = QLearner_ImageNN_funcs(num_actions, learning_rate)

        dummy_state = jnp.zeros((1, *self.image_size))
        self.params = self.funcs.f_qfunc.init(jax.random.PRNGKey(42), dummy_state)
        self.opt_state = self.funcs.f_opt.init(self.params)

        # target params for the network
        self.target_params = copy.deepcopy(self.params)

    def get_action_values(self, x: npt.ArrayLike) -> np.ndarray:
        action_values = self.funcs.f_get_q_values(self.params, x)
        return action_values

    def get_target_action_values(self, x: npt.ArrayLike) -> np.ndarray:
        action_values = self.funcs.f_get_q_values(self.target_params, x)
        return action_values
    
    def get_td_errors(self, data):
        return self.funcs.f_get_td_errors(self.params, self.target_params, data)

    def shift_update(self, data):
        self.params, self.opt_state = self.funcs.f_shift_update(self.params, self.opt_state, data)

    def update(self, data, polyak_stepsize:float=0.005, update_type: Literal['None', 'base_target']='None'):
        if update_type == 'None':
            update_func = self.funcs.f_update
        elif update_type == 'base_target':
            update_func = self.funcs.f_update_target
        else:
            raise NotImplementedError()

        self.params, self.opt_state, td_errors = update_func(self.params, self.target_params, self.opt_state, data)
        self.target_params = incremental_update(self.params, self.target_params, polyak_stepsize)
        return self.params, td_errors

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
        # globals.collector.collect('Q', np.copy(self.Q))   
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