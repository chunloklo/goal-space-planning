import haiku as hk
import jax
import optax
import jax.numpy as jnp
from typing import Any
import numpy.typing as npt
import copy
import numpy as np
from src.utils import globals

class QLearner_funcs():
    def __init__(self, num_actions: int, learning_rate: float):
        self.num_actions = num_actions

        def q_function(states):
            mlp = hk.Sequential([
                hk.Linear(128), jax.nn.relu,
                hk.Linear(128), jax.nn.relu,
                hk.Linear(64), jax.nn.relu,
                hk.Linear(64), jax.nn.relu,
                hk.Linear(self.num_actions),
            ])
            return mlp(states) 
        self.f_qfunc = hk.without_apply_rng(hk.transform(q_function))
        self.f_opt = optax.adam(learning_rate)

        def get_q_values(params: hk.Params, x: Any):
            action_values = self.f_qfunc.apply(params, x)
            return action_values
        
        def _take_action_index(data, action):
            return jnp.take_along_axis(data, jnp.expand_dims(action, axis=1), axis=1).squeeze()

        def get_td_errors(params: hk.Params, target_params: hk.Params, data):
            r = data['r']
            x = data['x']
            a = data['a']
            xp = data['xp']
            gamma = data['gamma']

            x_pred = self.f_qfunc.apply(params, x)
            # xp_pred = jax.lax.stop_gradient(jnp.max(self.f_qfunc.apply(target_params, xp), axis=1))
            xp_action_values = jax.lax.stop_gradient(self.f_qfunc.apply(target_params, xp))
            ap = jnp.argmax(self.f_qfunc.apply(params, xp), axis=1)
            xp_pred = _take_action_index(xp_action_values, ap)
            prev_pred = _take_action_index(x_pred, a)
            td_error = r + gamma * xp_pred - prev_pred
            return td_error
            
        def loss(params: hk.Params, target_params: hk.Params, data):
            td_errors = get_td_errors(params, target_params, data)
            return  jnp.mean(jnp.square(td_errors)), td_errors

        def update(params: hk.Params, target_params: hk.Params, opt_state, data):
            grads, td_errors = jax.grad(loss, has_aux=True)(params, target_params, data)
            updates, opt_state = self.f_opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, td_errors

        #### OCI

        def OCI_loss(params: hk.Params, target_params: hk.Params, data):
            # xp_goal_target = data['xp_goal_target']
            # x = data['x']
            # a = data['a']   
            # r = data['r']
            # xp = data['xp']
            # gamma = data['gamma']

            # x_pred = _take_action_index(self.f_qfunc.apply(params, x), a)
            # xp_action_values = jax.lax.stop_gradient(self.f_qfunc.apply(params, xp))

            # ap = jnp.argmax(jnp.maximum(xp_action_values, xp_goal_target), axis=1)
            # # print(ap.shape)

            # # Grabbing next update
            # xp_target = jax.lax.stop_gradient(self.f_qfunc.apply(target_params, xp))
            # target = r + gamma * _take_action_index(jnp.maximum(xp_target, xp_goal_target), ap)

            x = data['x']
            target = data['target']
            x_pred = self.f_qfunc.apply(params, x)

            return jnp.mean(jnp.square(target - x_pred))
        
        def OCI_combined_loss(params: hk.Params, target_params: hk.Params, data):
            r = data['r']
            x = data['x']
            a = data['a']
            xp = data['xp']
            gamma = data['gamma']
            goal_target = data['target']

            x_pred = self.f_qfunc.apply(params, x)
            xp_pred = jax.lax.stop_gradient(jnp.max(self.f_qfunc.apply(target_params, xp), axis=1))
            prev_pred = jnp.take_along_axis(x_pred, jnp.expand_dims(a, axis=1), axis=1).squeeze()

            goal_target_max_action = jnp.take_along_axis(goal_target, jnp.expand_dims(a, axis=1), axis=1).squeeze()
            print(goal_target_max_action.shape)

            td_error = jnp.maximum(r + gamma * xp_pred, goal_target_max_action) - prev_pred
            return  jnp.mean(jnp.square(td_error))

        def OCI_update(params: hk.Params, target_params: hk.Params, opt_state, data):
            loss, grads = jax.value_and_grad(OCI_loss)(params, target_params, data)
            updates, opt_state = self.f_opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss
        
        def OCI_combined_update(params: hk.Params, target_params: hk.Params, opt_state, data):
            grads = jax.grad(OCI_combined_loss)(params, target_params, data)
            updates, opt_state = self.f_opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state

        self.f_get_q_values = jax.jit(get_q_values)
        self.f_update = jax.jit(update)  
        self.f_get_td_errors = jax.jit(get_td_errors)   

        self.f_OCI_update = jax.jit(OCI_update)
        # self.f_OCI_update = OCI_update
        self.f_OCI_combined_update = jax.jit(OCI_combined_update)   
        return

class QLearner_NN():
    def __init__(self, state_shape, num_actions: int, learning_rate: float):
        self.num_actions: int = num_actions
        self.funcs = QLearner_funcs(num_actions, learning_rate)

        dummy_state = jnp.zeros(state_shape)
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

    def update(self, data, polyak_stepsize:float=0.005):
        self.params, self.opt_state, td_errors = self.funcs.f_update(self.params, self.target_params, self.opt_state, data)
        self.target_params = optax.incremental_update(self.params, self.target_params, polyak_stepsize)
        return self.params, td_errors

    def OCI_update(self, data, polyak_stepsize:float=0.005):
        self.params, self.opt_state, OCI_loss = self.funcs.f_OCI_update(self.params, self.target_params, self.opt_state, data)
        self.target_params = optax.incremental_update(self.params, self.target_params, polyak_stepsize)

        globals.collector.collect('OCI_loss', np.copy(OCI_loss))
        return self.params
    
    def OCI_combined_update(self, data, polyak_stepsize:float=0.005):
        self.params, self.opt_state = self.funcs.f_OCI_combined_update(self.params, self.target_params, self.opt_state, data)
        self.target_params = optax.incremental_update(self.params, self.target_params, polyak_stepsize)
        return self.params