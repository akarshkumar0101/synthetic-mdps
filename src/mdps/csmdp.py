import jax.numpy as jnp
import jax.random
from einops import rearrange
from jax.random import split

from .random_net import RandomMLP, create_random_net
from .smdp import Discrete, Box


def clip_state(state):
    # d_state = state.shape[-1]
    # max_norm = 2. + jnp.sqrt(d_state)
    #
    # norm = jnp.linalg.norm(state, axis=-1, keepdims=True)
    # clip_norm = norm.clip(min=None, max=max_norm)
    # state = state / norm * clip_norm
    # return state
    return state.clip(min=-3., max=3.)


class Init:
    def __init__(self, d_state, std=0.):
        self.d_state, self.std = d_state, std

    def sample_params(self, rng):
        mean = jax.random.normal(rng, (self.d_state,))
        params = dict(mean=mean)
        return params

    def __call__(self, rng, params):
        noise = jax.random.normal(rng, (self.d_state,))
        state = params['mean'] + noise * self.std
        state = clip_state(state)
        return state

    def state_space(self, params):
        return Box(-3, 3, (self.d_state,), dtype=jnp.float32)


class DeltaTransition:
    def __init__(self, d_state, n_acts, n_layers, d_hidden, activation, locality=1e-1, std=0.):
        self.d_state, self.n_acts = d_state, n_acts
        self.n_layers, self.d_hidden, self.activation = n_layers, d_hidden, activation
        self.locality, self.std = locality, std

        self.net = RandomMLP(n_layers=self.n_layers, d_hidden=self.d_hidden,
                             d_out=self.d_state * self.n_acts, activation=self.activation)

    def sample_params(self, rng):
        rng, _rng = split(rng)
        x = jax.random.normal(_rng, (32, self.d_state))
        rng, _rng = split(rng)
        params = create_random_net(self.net, _rng, x)
        return params

    def __call__(self, rng, state, action, params):
        mean = self.net.apply(params, state)
        mean = rearrange(mean, "(a d) -> a d", a=self.n_acts)[action]
        noise = jax.random.normal(rng, (self.d_state,))
        delta = mean * self.locality + noise * self.locality * self.std
        state = state + delta
        state = clip_state(state)
        return state

    def action_space(self, params):
        return Discrete(self.n_acts)


class Observation:
    def __init__(self, d_state, d_obs, n_layers, d_hidden, activation, std=0.):
        self.d_state, self.d_obs = d_state, d_obs
        self.n_layers, self.d_hidden, self.activation = n_layers, d_hidden, activation
        self.std = std

        self.net = RandomMLP(n_layers=self.n_layers, d_hidden=self.d_hidden,
                             d_out=self.d_obs, activation=self.activation)

    def sample_params(self, rng):
        rng, _rng = split(rng)
        x = jax.random.normal(_rng, (32, self.d_state))
        rng, _rng = split(rng)
        params = create_random_net(self.net, _rng, x)
        return params

    def __call__(self, rng, state, params):
        mean = self.net.apply(params, state)
        noise = jax.random.normal(rng, (self.d_obs,))
        obs = mean + noise * self.std
        return obs

    def observation_space(self, params):
        return Box(-3, 3, (self.d_obs,), dtype=jnp.float32)


class DenseReward:
    def __init__(self, d_state, n_layers, d_hidden, activation, std=0.):
        self.d_state = d_state
        self.n_layers, self.d_hidden, self.activation = n_layers, d_hidden, activation
        self.std = std
        self.net = RandomMLP(n_layers=self.n_layers, d_hidden=self.d_hidden,
                             d_out=1, activation=self.activation)

    def sample_params(self, rng):
        rng, _rng = split(rng)
        x = jax.random.normal(_rng, (32, self.d_state))
        rng, _rng = split(rng)
        params = create_random_net(self.net, _rng, x)
        return params

    def __call__(self, rng, state, params):
        mean = self.net.apply(params, state)
        noise = jax.random.normal(rng, (1,))
        rew = mean + noise * self.std
        return rew[..., 0]

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
