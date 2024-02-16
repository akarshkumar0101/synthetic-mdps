import jax
import jax.numpy as jnp

from .smdp import Discrete, Box


class Init:
    def __init__(self, n_states, std=0.):
        self.n_states, self.std = n_states, std

    def sample_params(self, rng):
        logits = jax.random.normal(rng, (self.n_states,)) * self.std
        params = dict(logits=logits)
        return params

    def __call__(self, rng, params):
        return jax.random.categorical(rng, params['logits'])

    def state_space(self, params):
        return Discrete(self.n_states)


class Transition:
    def __init__(self, n_states, n_acts):
        self.n_states, self.n_acts = n_states, n_acts

    def sample_params(self, rng):
        trans_matrix = jax.random.normal(rng, (self.n_acts, self.n_states, self.n_states))
        params = dict(trans_matrix=trans_matrix)
        return params

    def __call__(self, rng, state, action, params):
        trans_matrix = params['trans_matrix']
        logits = trans_matrix[action, :, state]
        state_n = jax.random.categorical(rng, logits)
        return state_n

    def action_space(self, params):
        return Discrete(self.n_acts)


class Observation:
    def __init__(self, n_states, d_obs):
        self.n_states, self.d_obs = n_states, d_obs

    def sample_params(self, rng):
        obs_matrix = jax.random.normal(rng, (self.n_states, self.d_obs))
        params = dict(obs_matrix=obs_matrix)
        return params

    def __call__(self, rng, state, params):
        obs_matrix = params['obs_matrix']
        return obs_matrix[state]

    def observation_space(self, params):
        return Box(-3, 3, (self.d_obs,), dtype=jnp.float32)


class DenseReward:
    def __init__(self, n_states):
        self.n_states = n_states

    def sample_params(self, rng):
        rew_matrix = jax.random.normal(rng, (self.n_states,))
        params = dict(rew_matrix=rew_matrix)
        return params

    def __call__(self, rng, state, params):
        return params['rew_matrix'][state]
