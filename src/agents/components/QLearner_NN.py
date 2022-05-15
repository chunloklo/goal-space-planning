import haiku as hk
import jax
import optax
import jax.numpy as jnp
from typing import Any
import numpy.typing as npt
import copy
import numpy as np
from src.utils import globals
from functools import partial


class QLearner_NN():
    def __init__(self, state_shape, num_actions: int, learning_rate: float, polyak_stepsize: float, beta, arch_flag, num_options: int = 0):
        self.num_actions = num_actions
        self.num_options = num_options
        self.polyak_stepsize = polyak_stepsize
        self.beta = beta

        assert arch_flag in [
            'pinball_simple',
            'pinball_simple_experimental',
            'pinball_hard'
        ]

        # Initializing jax functions
        init = hk.initializers.VarianceScaling(np.sqrt(2), 'fan_avg', 'uniform')
        b_init = hk.initializers.Constant(0.001)

        if arch_flag == 'pinball_simple':
            def q_function(states):
                mlp = hk.Sequential([
                    hk.Linear(128), jax.nn.relu,
                    hk.Linear(128), jax.nn.relu,
                    hk.Linear(64), jax.nn.relu,
                    hk.Linear(64), jax.nn.relu,
                    hk.Linear(self.num_actions + self.num_options),
                ])
                return mlp(states) 
        elif arch_flag == 'pinball_simple_experimental':
            def q_function(states):
                mlp = hk.Sequential([
                    hk.Linear(128, w_init = init, b_init = b_init), jax.nn.relu,
                    hk.Linear(128, w_init = init, b_init = b_init), jax.nn.relu,
                    hk.Linear(64, w_init = init, b_init = b_init), jax.nn.relu,
                    hk.Linear(self.num_actions + self.num_options),
                ])
                return mlp(states) 
        elif arch_flag == 'pinball_hard':
            def q_function(states):
                mlp = hk.Sequential([
                    hk.Linear(256, w_init = init, b_init = b_init), jax.nn.relu,
                    hk.Linear(256, w_init = init, b_init = b_init), jax.nn.relu,
                    hk.Linear(128, w_init = init, b_init = b_init), jax.nn.relu,
                    hk.Linear(128, w_init = init, b_init = b_init), jax.nn.relu,
                    hk.Linear(64, w_init = init, b_init = b_init), jax.nn.relu,
                    hk.Linear(64, w_init = init, b_init = b_init), jax.nn.relu,
                    hk.Linear(self.num_actions + self.num_options),
                ])
                return mlp(states) 
        else:
            raise NotImplementedError()

        self.network = hk.without_apply_rng(hk.transform(q_function))
        self.opt = optax.adam(learning_rate)

        dummy_state = jnp.zeros(state_shape)
        self.params = self.network.init(jax.random.PRNGKey(42), dummy_state)
        self.opt_state = self.opt.init(self.params)
        self.target_params = copy.deepcopy(self.params)
            

    def _take_action_index(self, data, action):
        return jnp.take_along_axis(data, jnp.expand_dims(action, axis=1), axis=1).squeeze()

    @partial(jax.jit, static_argnums=0)
    def _get_action_values(self, params: hk.Params, x: Any):
        action_values = self.network.apply(params, x)
        return action_values

    def get_action_values(self, x):
        return self._get_action_values(self.params, x)

    def _loss(self, params, target_params, data):
        r = data['r']
        x = data['x']
        a = data['a']
        xp = data['xp']
        gamma = data['gamma']

        x_pred = self.network.apply(params, x)
        # xp_pred = jax.lax.stop_gradient(jnp.max(self.f_qfunc.apply(target_params, xp), axis=1))
        xp_action_values = jax.lax.stop_gradient(self.network.apply(target_params, xp))
        ap = jnp.argmax(self.network.apply(params, xp), axis=1)
        xp_pred = self._take_action_index(xp_action_values, ap)
        prev_pred = self._take_action_index(x_pred, a)
        td_error = r + gamma * xp_pred - prev_pred

        loss = 0.5 * td_error**2

        return loss.mean()

    @partial(jax.jit, static_argnums=0)
    def _update(self, params: hk.Params, target_params: hk.Params, opt_state, polyak_stepsize, data):
        delta, grad = jax.value_and_grad(self._loss)(params, target_params, data)

        updates, opt_state = self.opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

        target_params = optax.incremental_update(params, target_params, polyak_stepsize)

        return params, target_params, opt_state, jnp.sqrt(delta)

    def update(self, data):
        self.params, self.target_params, self.opt_state, delta = self._update(self.params, self.target_params, self.opt_state, self.polyak_stepsize, data)
        return delta

    def _oci_loss(self, params, data):
        x = data['x']
        target = data['target']
        x_pred = self.network.apply(params, x)

        loss = 0.5 * (target - x_pred)**2

        return loss.mean()

    @partial(jax.jit, static_argnums=0)
    def _oci_update(self, params: hk.Params, opt_state, data):
        delta, grad = jax.value_and_grad(self._oci_loss)(params, data)

        updates, opt_state = self.opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

        return params, opt_state, jnp.sqrt(delta)

    def oci_update(self, data):
        self.params, self.opt_state, delta = self._oci_update(self.params, self.opt_state, data)
        return delta

    def _oci_target_loss(self, params, target_params, beta, data):
        r = data['r']
        x = data['x']
        a = data['a']
        xp = data['xp']
        gamma = data['gamma']
        target = data['target']

        x_pred = self.network.apply(params, x)
        xp_action_values = jax.lax.stop_gradient(self.network.apply(target_params, xp))
        ap = jnp.argmax(self.network.apply(params, xp), axis=1)
        xp_pred = self._take_action_index(xp_action_values, ap)
        prev_pred = self._take_action_index(x_pred, a)
        td_error = r + gamma * (beta * jnp.where(jnp.isnan(target), xp_pred, target) + (1 - beta) * xp_pred) - prev_pred
        loss = 0.5 * td_error**2

        return loss.mean()

    @partial(jax.jit, static_argnums=0)
    def _oci_target_update(self, params: hk.Params, target_params: hk.Params, opt_state, polyak_stepsize, beta, data):
        delta, grad = jax.value_and_grad(self._oci_target_loss)(params, target_params, beta, data)

        updates, opt_state = self.opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

        target_params = optax.incremental_update(params, target_params, polyak_stepsize)

        return params, target_params, opt_state, jnp.sqrt(delta)

    def oci_target_update(self, data):
        self.params, self.target_params, self.opt_state, delta = self._oci_target_update(self.params, self.target_params, self.opt_state, self.polyak_stepsize, self.beta, data)
        return delta

    
    def _option_value_loss(self, params, target_params, data):
        x = data['x']
        valid_goals = data['goal_inits']
        goal_values = data['goal_values']

        x_pred = self.network.apply(params, x)

        error = jnp.where(valid_goals == True, x_pred[self.num_actions:] - goal_values, 0)
        loss = 0.5 * error**2

        return loss.mean() + self._loss(params, target_params, data)

    @partial(jax.jit, static_argnums=0)
    def _option_value_update(self, params: hk.Params, target_params: hk.Params, opt_state, polyak_stepsize, beta, data):
        delta, grad = jax.value_and_grad(self._option_value_loss)(params, target_params, beta, data)

        updates, opt_state = self.opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

        target_params = optax.incremental_update(params, target_params, polyak_stepsize)

        return params, target_params, opt_state, jnp.sqrt(delta)
    
    def option_value_update(self, data):
        self.params, self.target_params, self.opt_state, delta = self._option_value_update(self.params, self.target_params, self.opt_state, self.polyak_stepsize, self.beta, data)
        return delta
    

