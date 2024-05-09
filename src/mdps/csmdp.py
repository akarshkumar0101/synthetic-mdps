import jax.numpy as jnp
import jax.random
from einops import rearrange
from jax.random import split

from .random_net import RandomMLP, create_random_net_normal
from .smdp import Discrete, Box


def constrain_state(state, constraint='clip', embeddings=None):
    if constraint == 'clip':
        return state.clip(min=-3., max=3.)
    elif constraint == 'ball': # not used
        d_state = state.shape[-1]
        max_norm = 2. + jnp.sqrt(d_state)
        norm = jnp.linalg.norm(state, axis=-1, keepdims=True)
        clip_norm = norm.clip(min=None, max=max_norm)
        state = state / norm * clip_norm
        return state
    elif constraint == 'unit_norm':
        d_state = state.shape[-1]
        return state / jnp.linalg.norm(state, axis=-1, keepdims=True) * jnp.sqrt(d_state)
    elif constraint == "embeddings":
        # return the closest embedding
        # state: d, embeddings: n, d
        idx = jnp.argmin(jnp.linalg.norm(embeddings-state, axis=-1))
        return embeddings[idx]
    else:
        raise NotImplementedError

class Init:
    def __init__(self, d_state, std=0., constraint='clip', n_embeds=None):
        self.d_state, self.std = d_state, std
        self.constraint = constraint
        self.n_embeds = n_embeds

    def sample_params(self, rng):
        mean = jax.random.normal(rng, (self.d_state,))
        params = {"init/mean": mean, "init/embeddings": None}
        if self.constraint == "embeddings":
            params["init/embeddings"] = jax.random.normal(rng, (self.n_embeds, self.d_state))
        return params

    def __call__(self, rng, params):
        state = params['init/mean'] + self.std * jax.random.normal(rng, (self.d_state,))
        return constrain_state(state, self.constraint, params['init/embeddings'])

    # def state_space(self, params):
    #     return Box(-3, 3, (self.d_state,), dtype=jnp.float32)


class Transition:
    def __init__(self, d_state, n_acts, std=0., locality=1e-1, constraint='clip', n_layers=0, d_hidden=16, activation=jax.nn.gelu):
        self.d_state, self.n_acts = d_state, n_acts
        self.std, self.locality, self.constraint = std, locality, constraint
        self.net = RandomMLP(n_layers=n_layers, d_hidden=d_hidden, d_out=self.d_state * self.n_acts, activation=activation)

    def sample_params(self, rng):
        net_params = create_random_net_normal(rng, self.net, batch_size=16, d_in=self.d_state)
        return {"trans/net_params": net_params}

    def __call__(self, rng, state, action, params):
        x =  self.net.apply(params['trans/net_params'], state)
        x = rearrange(x, "(a d) -> a d", a=self.n_acts)[action]
        x = x + self.std * jax.random.normal(rng, (self.d_state,))
        state = state + self.locality * x
        return constrain_state(state, self.constraint, params['init/embeddings'])


class Observation:
    def __init__(self, d_state, d_obs, std=0., n_layers=0, d_hidden=16, activation=jax.nn.gelu):
        self.d_state, self.d_obs, self.std = d_state, d_obs, std
        self.net = RandomMLP(n_layers=n_layers, d_hidden=d_hidden, d_out=self.d_obs, activation=activation)

    def sample_params(self, rng):
        net_params = create_random_net_normal(rng, self.net, batch_size=16, d_in=self.d_state)
        return {"obs/net_params": net_params}

    def __call__(self, rng, state, params):
        return self.net.apply(params['obs/net_params'], state) + self.std * jax.random.normal(rng, (self.d_obs,))

    # def observation_space(self, params):
    #     return Box(-3, 3, (self.d_obs,), dtype=jnp.float32)


class Reward:
    def __init__(self, d_state, std=0., sparse=False, sparse_prob=0.1,
                 n_layers=0, d_hidden=16, activation=jax.nn.gelu):
        self.d_state, self.std = d_state, std
        self.sparse, self.sparse_prob = sparse, sparse_prob
        self.net = RandomMLP(n_layers=n_layers, d_hidden=d_hidden, d_out=1, activation=activation)

    def sample_params(self, rng):
        net_params = create_random_net_normal(rng, self.net, batch_size=16, d_in=self.d_state)
        return {"rew/net_params": net_params}

    def __call__(self, rng, state, params):
        rew = self.net.apply(params['rew/net_params'], state) + self.std * jax.random.normal(rng, ())
        rew = rew[..., 0]
        if self.sparse:
            thresh = jax.scipy.stats.norm.ppf(self.sparse_prob)
            return (rew<thresh).astype(jnp.float32)
        else:
            return rew

class Done:
    def __init__(self, d_state, std=0., sparse_prob=0.,
                 n_layers=0, d_hidden=16, activation=jax.nn.gelu):
        self.d_state, self.std = d_state, std
        self.sparse_prob = sparse_prob
        self.net = RandomMLP(n_layers=n_layers, d_hidden=d_hidden, d_out=1, activation=activation)

    def sample_params(self, rng):
        net_params = create_random_net_normal(rng, self.net, batch_size=16, d_in=self.d_state)
        return {"done/net_params": net_params}

    def __call__(self, rng, state, params):
        done = self.net.apply(params['done/net_params'], state) + self.std * jax.random.normal(rng, ())
        done = done[..., 0]
        thresh = jax.scipy.stats.norm.ppf(self.sparse_prob)
        return done<thresh



# class DenseReward:
#     def __init__(self, d_state, n_layers, d_hidden, activation, std=0.):
#         self.d_state = d_state
#         self.n_layers, self.d_hidden, self.activation = n_layers, d_hidden, activation
#         self.std = std
#         self.net = RandomMLP(n_layers=self.n_layers, d_hidden=self.d_hidden,
#                              d_out=1, activation=self.activation)

#     def sample_params(self, rng):
#         rng, _rng = split(rng)
#         x = jax.random.normal(_rng, (32, self.d_state))
#         rng, _rng = split(rng)
#         params = create_random_net(self.net, _rng, x)
#         return params

#     def __call__(self, rng, state, params):
#         mean = self.net.apply(params, state)
#         noise = jax.random.normal(rng, (1,))
#         rew = mean + noise * self.std
#         return rew[..., 0]

# class GoalReward(nn.Module):
#     d_state: int
#     dist_thresh: float = None
#
#     def setup(self):
#         if self.dist_thresh is None:
#             # 4*(r**n)/2**n = .1
#             # 4*(r**n) = .1 * 2**n
#             # (r**n) = .1 * 2**n / 4
#             # r = (.1 * 2**n / 4)**(1/n)
#             # ensures ~10% of the state space is within the goal
#             self.r = (.1 * (2 ** self.d_state) / 4.) ** (1 / self.d_state)
#         bss_init = lambda rng, shape, dtype=None: jax.random.uniform(rng, shape, dtype=dtype, minval=-1., maxval=1.)
#         self.state_goal = self.param('state_goal', bss_init, (self.d_state,))
#
#     def __call__(self, state):
#         dist = jnp.linalg.norm(state - self.state_goal)
#         rew = (dist < self.r).astype(jnp.float32)
#         return rew
