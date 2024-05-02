import jax.numpy as jnp
import jax.random
from einops import rearrange
from jax.random import split

from .random_net import RandomMLP, create_random_net
from .smdp import Discrete, Box


def constrain_state(state, embeddings):
    # return the closest embedding
    # state: d, embeddings: n, d
    idx = jnp.argmin(jnp.linalg.norm(embeddings-state, axis=-1))
    return embeddings[idx]


class Transition:
    def __init__(self, d_state, n_acts, std=0., locality=1e-1, constraint='clip', n_layers=0, d_hidden=16, activation=jax.nn.gelu):
        self.d_state, self.n_acts = d_state, n_acts
        self.std, self.locality, self.constraint = std, locality, constraint
        self.net = RandomMLP(n_layers=n_layers, d_hidden=d_hidden, d_out=self.d_state * self.n_acts, activation=activation)

    def sample_params(self, rng):
        _rng1, _rng2 = split(rng)
        embeddings = jax.random.normal(_rng1, (self.n_embeds, self.d_state))
        net_params = create_random_net_normal(_rng2, self.net, batch_size=16, d_in=self.d_state)
        return dict(net_params=net_params, embedding=embeddings)

    def __call__(self, rng, state, action, params):
        x =  self.net.apply(params['net_params'], state)
        x = rearrange(x, "(a d) -> a d", a=self.n_acts)[action]
        x = x + self.std * jax.random.normal(rng, (self.d_state,))
        state = state + self.locality * x
        return constrain_state(state, params['embedding'])

    # def action_space(self, params):
    #     return Discrete(self.n_acts)

