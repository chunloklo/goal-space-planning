import haiku as hk
import jax
import jax.numpy as jnp
import optax
import numpy as np
import numpy.typing as npt
import copy
from typing import Any
from functools import partial

class EQRC_NN():
    def __init__(self, state_shape, num_actions: int, step_size: float, epsilon: float, beta: float = 1.0):
        self.num_actions: int = num_actions
        self.epsilon = epsilon
        self.beta = beta

        # Initialization
        dummy_state = jnp.zeros(state_shape)

        # Initializing jax functions
        init = hk.initializers.VarianceScaling(np.sqrt(2), 'fan_avg', 'uniform')
        b_init = hk.initializers.Constant(0)

        def network(states):
            hidden = hk.Sequential([
                hk.Linear(128, w_init=init, b_init=b_init), jax.nn.relu,
                hk.Linear(128, w_init=init, b_init=b_init), jax.nn.relu,
                hk.Linear(64, w_init=init, b_init=b_init), jax.nn.relu,
                hk.Linear(64, w_init=init, b_init=b_init), jax.nn.relu,
                # hk.Linear(128, w_init=init, b_init=b_init), jax.nn.relu,
                # hk.Linear(128, w_init=init, b_init=b_init), jax.nn.relu,
            ])

            values = hk.Sequential([
                hk.Linear(self.num_actions, w_init=init, b_init=b_init)
            ])
                    
            h = hidden(states)
            v = values(h)

            return v, h

        self.network = hk.without_apply_rng(hk.transform(network))
        net_params = self.network.init(jax.random.PRNGKey(99), dummy_state)

        # to build the secondary weights, we need to know the size of the "feature layer" of our nn
        # there is almost certainly a better way than this, but it's fine
        _, x = self.network.apply(net_params, dummy_state)

        def h(x):
            h = hk.Sequential([
                hk.Linear(self.num_actions, w_init=hk.initializers.Constant(0), b_init=hk.initializers.Constant(0))
            ])
            return h(jax.lax.stop_gradient(x))
        
        self.h = hk.without_apply_rng(hk.transform(h))
        # Using the hidden layer output
        h_params = self.h.init(jax.random.PRNGKey(100), x)

        self.params = {
            'w': net_params,
            'h': h_params
        }

        self.f_opt = optax.adam(step_size)
        self.opt_state = self.f_opt.init(self.params)

    def _take_action_index(data, action):
        return jnp.take_along_axis(data, jnp.expand_dims(action, axis=1), axis=1).squeeze()

    def _loss(self, params: hk.Params, data):
        r = data['r']
        x = data['x']
        a = data['a']
        xp = data['xp']
        gamma = data['gamma']
        
        q, phi = self.network.apply(params['w'], x)
        qp, _ = self.network.apply(params['w'], xp)

        h = self.h.apply(params['h'], phi) 

        v_loss, h_loss = jax.vmap(partial(qc_loss, self.epsilon), in_axes=0)(q, a, r, gamma, qp, h)
        h_loss = h_loss.mean()
        v_loss = v_loss.mean()

        regularizer = sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params['h']))

        return v_loss + h_loss + self.beta * regularizer

    @partial(jax.jit, static_argnums=0)
    def _get_action_values(self, params: hk.Params, x: Any):
        q, phi = self.network.apply(params, x)
        return q

    @partial(jax.jit, static_argnums=0)
    def _update(self, params: hk.Params, opt_state, data):
        delta, grad = jax.value_and_grad(self._loss)(params, data)

        updates, opt_state = self.f_opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

        return params, opt_state, jnp.sqrt(delta)
    
    # Public facing getting goal outputs
    def get_action_values(self, x: npt.ArrayLike) -> np.ndarray:
        q = self._get_action_values(self.params['w'], x)
        return q

    def update(self, data):
        self.params, self.opt_state, delta = self._update(self.params, self.opt_state, data)
        return delta

def _argmax_with_random_tie_breaking(preferences):
    optimal_actions = (preferences == preferences.max(axis=-1, keepdims=True))
    return optimal_actions / optimal_actions.sum(axis=-1, keepdims=True)

def qc_loss(epsilon, x, a, r, gamma, xp, h):
    pi = _argmax_with_random_tie_breaking(xp)

    pi = (1.0 - epsilon) * pi + (epsilon / xp.shape[0])
    pi = jax.lax.stop_gradient(pi)

    vp = xp.dot(pi)
    target = r + gamma * vp
    target = jax.lax.stop_gradient(target)

    delta = target - x[a]
    delta_hat = h[a]

    v_loss = 0.5 * delta**2 + gamma * jax.lax.stop_gradient(delta_hat) * vp
    h_loss = 0.5 * (jax.lax.stop_gradient(delta) - delta_hat)**2

    return v_loss, h_loss