import haiku as hk
import jax
import jax.numpy as jnp
import optax
import numpy as np
import numpy.typing as npt
import copy
from typing import Any
from functools import partial

class GoalLearner_EQRC_NN():
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
                hk.Linear(128, w_init=init, b_init=b_init), jax.nn.relu,
                hk.Linear(128, w_init=init, b_init=b_init), jax.nn.relu,
                hk.Linear(128, w_init=init, b_init=b_init), jax.nn.relu,
                hk.Linear(128, w_init=init, b_init=b_init), jax.nn.relu,
            ])

            values = hk.Sequential([
                hk.Linear(self.num_actions, w_init=init, b_init=b_init)
            ])
                    
            h = hidden(states)
            v = values(h)

            return v, h

        self.network = hk.without_apply_rng(hk.transform(network))
        gamma_params = self.network.init(jax.random.PRNGKey(99), dummy_state)
        value_params = self.network.init(jax.random.PRNGKey(100), dummy_state)

        # to build the secondary weights, we need to know the size of the "feature layer" of our nn
        # there is almost certainly a better way than this, but it's fine
        _, x = self.network.apply(gamma_params, dummy_state)

        def h(x):
            h = hk.Sequential([
                hk.Linear(self.num_actions, w_init=hk.initializers.Constant(0), b_init=hk.initializers.Constant(0))
            ])
            return h(jax.lax.stop_gradient(x))
        
        self.h = hk.without_apply_rng(hk.transform(h))
        # Using the hidden layer output
        gamma_h_params = self.h.init(jax.random.PRNGKey(100), x)
        value_h_params = self.h.init(jax.random.PRNGKey(100), x)

        self.params = {
            'gamma_w': gamma_params,
            'gamma_h': gamma_h_params,
            'value_w': value_params,
            'value_h': value_h_params
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
        goal_policy_cumulant = data['goal_policy_cumulant']
        goal_discount = data['goal_discount']
        
        q, phi = self.network.apply(params['gamma_w'], x)
        qp, _ = self.network.apply(params['gamma_w'], xp)

        h = self.h.apply(params['gamma_h'], phi)

        pi_qp = _argmax_with_random_tie_breaking(qp)
        pi_qp = (1.0 - self.epsilon) * pi_qp + (self.epsilon / qp.shape[0])

        v_loss, h_loss = jax.vmap(qc_loss, in_axes=0)(q, a, goal_policy_cumulant, goal_discount, qp, h, pi_qp)
        h_loss = h_loss.mean()
        v_loss = v_loss.mean()

        regularizer = sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params['gamma_h']))

        # Gamma loss
        gamma_loss = v_loss + h_loss + self.beta * regularizer

        q, phi = self.network.apply(params['value_w'], x)
        qp, _ = self.network.apply(params['value_w'], xp)

        h = self.h.apply(params['value_h'], phi) 

        v_loss, h_loss = jax.vmap(qc_loss, in_axes=0)(q, a, r, goal_discount, qp, h, pi_qp)
        h_loss = h_loss.mean()
        v_loss = v_loss.mean()

        regularizer = sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params['value_h']))

        value_loss = v_loss + h_loss + self.beta * regularizer

        return gamma_loss + value_loss

    @partial(jax.jit, static_argnums=0)
    def _get_goal_output(self, params: hk.Params, x: Any):
        q, phi = self.network.apply(params, x)
        return q

    @partial(jax.jit, static_argnums=0)
    def _update(self, params: hk.Params, opt_state, data):
        delta, grad = jax.value_and_grad(self._loss)(params, data)

        updates, opt_state = self.f_opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

        return params, opt_state, jnp.sqrt(delta)
    
    # Public facing getting goal outputs
    def get_goal_outputs(self, x: npt.ArrayLike) -> np.ndarray:
        gamma = self._get_goal_output(self.params['gamma_w'], x)
        value = self._get_goal_output(self.params['value_w'], x)
        # Returning the same 3 thing for now.
        return gamma, value, gamma

    def update(self, data):
        self.params, self.opt_state, delta = self._update(self.params, self.opt_state, data)
        return delta

def _argmax_with_random_tie_breaking(preferences):
    optimal_actions = (preferences == preferences.max(axis=-1, keepdims=True))
    return optimal_actions / optimal_actions.sum(axis=-1, keepdims=True)

def qc_loss(q, a, r, gamma, qp, h, pi_qp):
    pi_qp = jax.lax.stop_gradient(pi_qp) 
    vp = qp.dot(pi_qp)
    target = r + gamma * vp
    target = jax.lax.stop_gradient(target)

    delta = target - q[a]
    delta_hat = h[a]

    v_loss = 0.5 * delta**2 + gamma * jax.lax.stop_gradient(delta_hat) * vp
    h_loss = 0.5 * (jax.lax.stop_gradient(delta) - delta_hat)**2

    return v_loss, h_loss
class GoalLearner_QRC_NN(GoalLearner_EQRC_NN):
    def __init__(self, state_shape, num_actions: int, step_size: float, beta: float = 1.0):
        super().__init__(state_shape, num_actions, step_size, 0, beta)
