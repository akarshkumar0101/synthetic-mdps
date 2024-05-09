import jax
import jax.numpy as jnp

from einops import rearrange
from .smdp import Discrete, Box

from .random_net import RandomMLP, create_random_net_normal

class Init:
    def __init__(self, m, n, temperature=1):
        self.m, self.n = m, n
        self.temperature = temperature

    def sample_params(self, rng):
        logits = jax.random.normal(rng, (self.m, self.n))
        return {"init/logits": logits}

    def __call__(self, rng, params):
        return jax.random.categorical(rng, params['init/logits']/self.temperature, axis=-1)


class Transition:
    def __init__(self, m, n, n_acts, temperature=1., n_layers=0, d_hidden=16, activation=jax.nn.gelu):
        self.m, self.n, self.n_acts, self.temperature = m, n, n_acts, temperature
        self.net = RandomMLP(n_layers=n_layers, d_hidden=d_hidden, d_out=self.m*self.n, activation=activation)

    def sample_params(self, rng):
        net_params = create_random_net_normal(rng, self.net, batch_size=16, d_in=self.m*self.n + self.n_acts)
        return {"trans/net_params": net_params}

    def __call__(self, rng, state, action, params): 
        x1 = jax.nn.one_hot(state, self.n).flatten() # (m, n)
        x2 = jax.nn.one_hot(action, self.n_acts)
        x = jnp.concatenate([x1, x2], axis=-1)*2.-1.

        logits = self.net.apply(params['trans/net_params'], x)
        logits = rearrange(logits, "(m n) -> m n", m=self.m)
        state_n = jax.random.categorical(rng, logits/self.temperature, axis=-1)
        return state_n

class Observation:
    def __init__(self, m, n, d_obs, std=0., n_layers=0, d_hidden=16, activation=jax.nn.gelu):
        self.m, self.n, self.d_obs = m, n, d_obs
        self.std = std

        self.net = RandomMLP(n_layers=n_layers, d_hidden=d_hidden, d_out=d_obs, activation=activation)

    def sample_params(self, rng):
        net_params = create_random_net_normal(rng, self.net, batch_size=16, d_in=self.m*self.n)
        return {"obs/net_params": net_params}

    def __call__(self, rng, state, params):
        x1 = jax.nn.one_hot(state, self.n).flatten() # (m, n)
        x = x1*2.-1.
        return self.net.apply(params['obs/net_params'], x) + self.std * jax.random.normal(rng, (self.d_obs,))

class Reward:
    def __init__(self, m, n, std=0., sparse=False, sparse_prob=0.1,
                 n_layers=0, d_hidden=16, activation=jax.nn.gelu):
        self.m, self.n, self.std = m, n, std
        self.sparse, self.sparse_prob = sparse, sparse_prob
        self.net = RandomMLP(n_layers=n_layers, d_hidden=d_hidden, d_out=1, activation=activation)

    def sample_params(self, rng):
        net_params = create_random_net_normal(rng, self.net, batch_size=16, d_in=self.m*self.n)
        return {"rew/net_params": net_params}

    def __call__(self, rng, state, params):
        x1 = jax.nn.one_hot(state, self.n).flatten() # (m, n)
        x = x1*2.-1.
        rew = self.net.apply(params['rew/net_params'], x) + self.std * jax.random.normal(rng, ())
        rew = rew[..., 0]
        if self.sparse:
            thresh = jax.scipy.stats.norm.ppf(self.sparse_prob)
            return (rew<thresh).astype(jnp.float32)
        else:
            return rew

class Done:
    def __init__(self, m, n, std=0., sparse_prob=0.,
                 n_layers=0, d_hidden=16, activation=jax.nn.gelu):
        self.m, self.n, self.std = m, n, std
        self.sparse_prob = sparse_prob
        self.net = RandomMLP(n_layers=n_layers, d_hidden=d_hidden, d_out=1, activation=activation)

    def sample_params(self, rng):
        net_params = create_random_net_normal(rng, self.net, batch_size=16, d_in=self.m*self.n)
        return {"done/net_params": net_params}

    def __call__(self, rng, state, params):
        x1 = jax.nn.one_hot(state, self.n).flatten() # (m, n)
        x = x1*2.-1.
        done = self.net.apply(params['done/net_params'], x) + self.std * jax.random.normal(rng, ())
        done = done[..., 0]
        thresh = jax.scipy.stats.norm.ppf(self.sparse_prob)
        return done<thresh
