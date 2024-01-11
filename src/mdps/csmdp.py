import flax.linen as nn
import jax.numpy as jnp
import jax.random
from einops import rearrange

from .smdp import Discrete, Box


class Init(nn.Module):
    d_state: int

    def setup(self):
        init = lambda rng, shape, dtype=None: jax.random.uniform(rng, shape, dtype=dtype, minval=-1., maxval=1.)
        self.state_start = self.param('state_start', init, (self.d_state,))

    def __call__(self, rng):
        state = self.state_start
        return state

    def state_space(self, params):
        return Box(-1, 1, (self.d_state,), dtype=jnp.float32)


class LinearTransition(nn.Module):
    d_state: int
    n_acts: int
    delta: True

    def setup(self):
        self.net = nn.Dense(self.d_state * self.n_acts)

    def __call__(self, rng, state, action):
        state_n = rearrange(self.net(state), "(a d) -> a d", a=self.n_acts)[action]
        if self.delta:
            state_n = state + .15 * state_n
        state_n = jnp.clip(state_n, -1., 1.)
        return state_n

    def action_space(self, params):
        return Discrete(self.n_acts)


class MLPTransition(nn.Module):
    d_state: int
    n_acts: int
    delta: True

    def setup(self):
        self.net = nn.Sequential([
            nn.Dense(self.d_state * self.n_acts,
                     kernel_init=jax.nn.initializers.orthogonal(),
                     bias_init=jax.nn.initializers.normal(1e-1)),
            nn.tanh,
            nn.Dense(self.d_state * self.n_acts,
                     kernel_init=jax.nn.initializers.orthogonal(),
                     bias_init=jax.nn.initializers.normal(1e-1)),
            nn.tanh,
            nn.Dense(self.d_state * self.n_acts),
        ])

    def __call__(self, rng, state, action):
        state_n = rearrange(self.net(state), "(a d) -> a d", a=self.n_acts)[action]
        if self.delta:
            state_n = state + .15 * state_n
        state_n = jnp.clip(state_n, -1., 1.)
        return state_n

    def action_space(self, params):
        return Discrete(self.n_acts)


class LinearObservation(nn.Module):
    d_state: int
    d_obs: int
    initializer: nn.initializers.Initializer = nn.initializers.normal(stddev=1.)

    def setup(self):
        self.seq = nn.Dense(self.d_obs, use_bias=True, kernel_init=self.initializer, bias_init=self.initializer)

    def __call__(self, state):
        return self.seq(state)

    def observation_space(self, params):
        return Box(-1, 1, (self.d_obs,), dtype=jnp.float32)


class LinearReward(nn.Module):
    d_state: int

    def setup(self):
        ball_init = lambda rng, shape, dtype=None: jax.random.ball(rng, d=shape[0], dtype=dtype)
        self.dir_goal = self.param('dir_goal', ball_init, (self.d_state,))

    def __call__(self, state):
        return jnp.dot(state, self.dir_goal)


class GoalReward(nn.Module):
    d_state: int
    dist_thresh: float = None

    def setup(self):
        if self.dist_thresh is None:
            # 4*(r**n)/2**n = .1
            # 4*(r**n) = .1 * 2**n
            # (r**n) = .1 * 2**n / 4
            # r = (.1 * 2**n / 4)**(1/n)
            # ensures ~10% of the state space is within the goal
            self.r = (.1 * (2 ** self.d_state) / 4.) ** (1 / self.d_state)
        bss_init = lambda rng, shape, dtype=None: jax.random.uniform(rng, shape, dtype=dtype, minval=-1., maxval=1.)
        self.state_goal = self.param('state_goal', bss_init, (self.d_state,))

    def __call__(self, state):
        dist = jnp.linalg.norm(state - self.state_goal)
        rew = (dist < self.r).astype(jnp.float32)
        return rew
