import jax
import jax.numpy as jnp

from .smdp import Discrete, Box


class Init:
    def __init__(self, m, n, temperature=1):
        self.m, self.n = m, n
        self.temperature = temperature

    def sample_params(self, rng):
        logits = jax.random.normal(rng, (self.m, self.n)) / self.temperature
        return dict(logits=logits)

    def __call__(self, rng, params):
        return jax.random.categorical(rng, params['logits'], axis=-1)


class Transition:
    def __init__(self, m, n, n_acts, temperature=0.):
        self.m, self.n = m, n
        self.n_acts = n_acts
        self.temperature = temperature

    def sample_params(self, rng):
        rng, _rng = split(rng)
        x = jax.random.normal(_rng, (32, self.m*self.n + self.n_acts))
        rng, _rng = split(rng)
        params = create_random_net(self.net, _rng, x)
        return params

    def __call__(self, rng, state, action, params): 
        x1 = jax.nn.one_hot(state, self.n).flatten() # (m, n)
        x2 = jax.nn.one_hot(action)
        x = jnp.concatenate([x1, x2], axis=-1)

        logits = self.net.apply(params, x)
        logits = rearrange(logits, "(m n) -> m n", m=self.m)
        state_n = jax.random.categorical(rng, logits/self.temperature, axis=-1)
        return state_n

class Observation:
    def __init__(self, n_states, d_obs, std=0.):
        self.n_states, self.d_obs = n_states, d_obs
        self.std = std

    def sample_params(self, rng):
        obs_matrix = jax.random.normal(rng, (self.n_states, self.d_obs))
        params = dict(obs_matrix=obs_matrix)
        return params

    def __call__(self, rng, state, params):
        mean = params['obs_matrix'][state]
        noise = jax.random.normal(rng, (self.d_obs,))
        obs = mean + noise * self.std
        return obs

    def observation_space(self, params):
        return Box(-3, 3, (self.d_obs,), dtype=jnp.float32)


class DenseReward:
    def __init__(self, n_states, std=0.):
        self.n_states = n_states
        self.std = std

    def sample_params(self, rng):
        rew_matrix = jax.random.normal(rng, (self.n_states, 1))
        params = dict(rew_matrix=rew_matrix)
        return params

    def __call__(self, rng, state, params):
        mean = params['rew_matrix'][state]
        noise = jax.random.normal(rng, (1,))
        rew = mean + noise * self.std
        return rew[..., 0]
