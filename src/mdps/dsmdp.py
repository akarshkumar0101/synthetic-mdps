import jax
import jax.numpy as jnp

from .smdp import Discrete, Box


class Init:
    def __init__(self, n, temperature=1.):
        self.n, self.temperature = n, temperature

    def sample_params(self, rng):
        logits = jax.random.normal(rng, (self.n,))
        return dict(logits=logits)

    def __call__(self, rng, params):
        return jax.random.categorical(rng, params['logits'] / self.temperature, axis=-1)

    # def state_space(self, params):
    #     return Discrete(self.n_states)


class Transition:
    def __init__(self, n, n_acts, temperature=1.):
        self.n, self.n_acts, self.temperature = n, n_acts, temperature

    def sample_params(self, rng):
        trans_matrix = jax.random.normal(rng, (self.n_acts, self.n, self.n))
        return dict(trans_matrix=trans_matrix)

    def __call__(self, rng, state, action, params):
        logits = params['trans_matrix'][action, state, :]
        state_n = jax.random.categorical(rng, logits / self.temperature, axis=-1)
        return state_n

    # def action_space(self, params):
    #     return Discrete(self.n_acts)


class Observation:
    def __init__(self, n, d_obs, std=0.):
        self.n, self.d_obs, self.std = n, d_obs, std

    def sample_params(self, rng):
        obs_matrix = jax.random.normal(rng, (self.n, self.d_obs))
        return dict(obs_matrix=obs_matrix)

    def __call__(self, rng, state, params):
        return params['obs_matrix'][state] + self.std * jax.random.normal(rng, (self.d_obs,))

    # def observation_space(self, params):
    #     return Box(-3, 3, (self.d_obs,), dtype=jnp.float32)


class Reward:
    def __init__(self, n, std=0., sparse=False, sparse_prob=0.1):
        self.n, self.std = n, std
        self.sparse, self.sparse_prob = sparse, sparse_prob

    def sample_params(self, rng):
        rew_matrix = jax.random.normal(rng, (self.n, ))
        return dict(rew_matrix=rew_matrix)

    def __call__(self, rng, state, params):
        rew = params['rew_matrix'][state] + self.std * jax.random.normal(rng, ())
        if self.sparse:
            thresh = jax.scipy.stats.norm.ppf(self.sparse_prob)
            return (rew<thresh).astype(jnp.float32)
        else:
            return rew

class Done:
    def __init__(self, n, std=0., sparse_prob=0.):
        self.n, self.std = n, std
        self.sparse_prob = sparse_prob

    def sample_params(self, rng):
        done_matrix = jax.random.normal(rng, (self.n, ))
        return dict(done_matrix=done_matrix)

    def __call__(self, rng, state, params):
        done = params['done_matrix'][state] + self.std * jax.random.normal(rng, ())
        thresh = jax.scipy.stats.norm.ppf(self.sparse_prob)
        return done<thresh
