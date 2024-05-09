
import jax.numpy as jnp
import jax.random
from einops import rearrange
from jax.random import split

from .random_net import RandomMLP, create_random_net_normal
from .smdp import Discrete, Box
from .csmdp import constrain_state


class Init:
    def __init__(self, m, n, d_state, temperature=1., std=0., constraint='clip', n_embeds=None):
        self.m, self.n = m, n
        self.temperature = temperature

        self.d_state, self.std = d_state, std
        self.constraint = constraint
        self.n_embeds = n_embeds

    def sample_params(self, rng):
        mean = jax.random.normal(rng, (self.d_state,))
        logits = jax.random.normal(rng, (self.m, self.n))
        params = {"init/logits": logits, "init/mean": mean, "init/embeddings": None}
        if self.constraint == "embeddings":
            params["init/embeddings"] = jax.random.normal(rng, (self.n_embeds, self.d_state))
        return params

    def __call__(self, rng, params):
        state1 = jax.random.categorical(rng, params['init/logits']/self.temperature, axis=-1)
        state2 = params['init/mean'] + self.std * jax.random.normal(rng, (self.d_state,))
        state2 = constrain_state(state2, self.constraint, params['init/embeddings'])
        return (state1, state2)


class Transition:
    def __init__(self, m, n, d_state, n_acts, temperature=1., std=0., locality=1e-1, constraint='clip', n_layers=0, d_hidden=16, activation=jax.nn.gelu):
        self.m, self.n, self.n_acts, self.temperature = m, n, n_acts, temperature
        self.d_state, self.n_acts = d_state, n_acts
        self.std, self.locality, self.constraint = std, locality, constraint
        self.net = RandomMLP(n_layers=n_layers, d_hidden=d_hidden, d_out=(self.m*self.n+self.d_state), activation=activation)

    def sample_params(self, rng):
        net_params = create_random_net_normal(rng, self.net, batch_size=16, d_in=(self.m*self.n+self.d_state+self.n_acts))
        return {"trans/net_params": net_params}

    def __call__(self, rng, state, action, params):
        state1, state2 = state

        x1 = jax.nn.one_hot(state1, self.n).flatten() # (m, n)
        x2 = jax.nn.one_hot(action, self.n_acts)
        x3 = state2
        x = jnp.concatenate([x1*2-1., x2*2-1., x3], axis=-1)
        x =  self.net.apply(params['trans/net_params'], x)

        logits, x = jnp.split(x, [self.m*self.n], axis=-1)
        logits = rearrange(logits, "(m n) -> m n", m=self.m)
        state1 = jax.random.categorical(rng, logits/self.temperature, axis=-1)
        x = x + self.std * jax.random.normal(rng, (self.d_state,))
        state2 = state2 + self.locality * x
        state2 = constrain_state(state2, self.constraint, params['init/embeddings'])
        return (state1, state2)


class Observation:
    def __init__(self, m, n, d_state, d_obs, std=0., n_layers=0, d_hidden=16, activation=jax.nn.gelu):
        self.m, self.n, self.d_obs = m, n, d_obs
        self.d_state, self.d_obs, self.std = d_state, d_obs, std
        self.net = RandomMLP(n_layers=n_layers, d_hidden=d_hidden, d_out=self.d_obs, activation=activation)

    def sample_params(self, rng):
        net_params = create_random_net_normal(rng, self.net, batch_size=16, d_in=(self.m*self.n+self.d_state))
        return {"obs/net_params": net_params}

    def __call__(self, rng, state, params):
        x1, x2 = state
        x1 = jax.nn.one_hot(x1, self.n).flatten() # (m, n)
        x = jnp.concatenate([x1*2-1., x2], axis=-1)
        return self.net.apply(params['obs/net_params'], x) + self.std * jax.random.normal(rng, (self.d_obs,))

class Reward:
    def __init__(self, m, n, d_state, std=0., sparse=False, sparse_prob=0.1,
                 n_layers=0, d_hidden=16, activation=jax.nn.gelu):
        self.m, self.n, self.std = m, n, std
        self.d_state, self.std = d_state, std
        self.sparse, self.sparse_prob = sparse, sparse_prob
        self.net = RandomMLP(n_layers=n_layers, d_hidden=d_hidden, d_out=1, activation=activation)

    def sample_params(self, rng):
        net_params = create_random_net_normal(rng, self.net, batch_size=16, d_in=(self.m*self.n+self.d_state))
        return {"rew/net_params": net_params}

    def __call__(self, rng, state, params):
        x1, x2 = state
        x1 = jax.nn.one_hot(x1, self.n).flatten() # (m, n)
        x = jnp.concatenate([x1*2-1., x2], axis=-1)
        rew = self.net.apply(params['rew/net_params'], x) + self.std * jax.random.normal(rng, ())
        rew = rew[..., 0]
        if self.sparse:
            thresh = jax.scipy.stats.norm.ppf(self.sparse_prob)
            return (rew<thresh).astype(jnp.float32)
        else:
            return rew

class Done:
    def __init__(self, m, n, d_state, std=0., sparse_prob=0.,
                 n_layers=0, d_hidden=16, activation=jax.nn.gelu):
        self.m, self.n, self.std = m, n, std
        self.d_state, self.std = d_state, std
        self.sparse_prob = sparse_prob
        self.net = RandomMLP(n_layers=n_layers, d_hidden=d_hidden, d_out=1, activation=activation)

    def sample_params(self, rng):
        net_params = create_random_net_normal(rng, self.net, batch_size=16, d_in=(self.m*self.n+self.d_state))
        return {"done/net_params": net_params}

    def __call__(self, rng, state, params):
        x1, x2 = state
        x1 = jax.nn.one_hot(x1, self.n).flatten() # (m, n)
        x = jnp.concatenate([x1*2-1., x2], axis=-1)
        done = self.net.apply(params['done/net_params'], x) + self.std * jax.random.normal(rng, ())
        done = done[..., 0]
        thresh = jax.scipy.stats.norm.ppf(self.sparse_prob)
        return done<thresh



