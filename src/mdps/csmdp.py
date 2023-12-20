import flax.linen as nn
import jax.numpy as jnp
import jax.random
from gymnax.environments import spaces


class Init(nn.Module):
    d_state: int
    initializer: nn.initializers.Initializer = nn.initializers.normal(stddev=1.)
    dist_thresh: float = 3e-1

    def setup(self):
        self.state_start = self.param('state_start', self.initializer, (self.d_state,))

    def __call__(self, rng):
        state = self.state_start + self.dist_thresh * jax.random.ball(rng, d=self.d_state)
        return state


class Transition1(nn.Module):
    d_state: int
    n_acts: int
    delta: True
    clip: int = 3.

    def setup(self):
        self.embed_act = nn.Embed(self.n_acts, self.d_state)
        self.seq = nn.Sequential([
            nn.Dense(self.d_state),
            nn.tanh,
            nn.Dense(self.d_state),
            nn.tanh,
            nn.Dense(self.d_state),
        ])

    def __call__(self, rng, state, action):
        x = jnp.concatenate([state, self.embed_act(action)], axis=-1)
        x = 1e-1 * self.seq(x)
        state_n = state + x if self.delta else x
        state_n = jnp.clip(state_n, -self.clip, self.clip)
        return state_n

    def action_space(self, params):
        return spaces.Discrete(self.n_acts)


class Transition2(nn.Module):
    d_state: int
    n_acts: int
    delta: True
    clip: int = 3.

    def setup(self):
        self.embed_act = nn.Embed(self.n_acts, self.d_state)
        self.seq = nn.Sequential([
            nn.Dense(self.d_state*4,
                     kernel_init=jax.nn.initializers.orthogonal(),
                     bias_init=jax.nn.initializers.normal(1.)),
            nn.tanh,
            nn.Dense(self.d_state*4,
                     kernel_init=jax.nn.initializers.orthogonal(),
                     bias_init=jax.nn.initializers.normal(1e-2)),
            nn.tanh,
            nn.Dense(self.d_state*4,
                     kernel_init=jax.nn.initializers.orthogonal(),
                     bias_init=jax.nn.initializers.normal(1e-2)),
            nn.tanh,
            nn.Dense(self.d_state*4,
                     kernel_init=jax.nn.initializers.orthogonal(),
                     bias_init=jax.nn.initializers.normal(1e-2)),
            nn.tanh,
            nn.Dense(self.n_acts * self.d_state),

        ])

    def __call__(self, rng, state, action):
        state_n = 0.3*self.seq(state).reshape(self.n_acts, self.d_state)[action]
        if self.delta:
            state_n = state + state_n
        state_n = jnp.clip(state_n, -self.clip, self.clip)
        return state_n

    def action_space(self, params):
        return spaces.Discrete(self.n_acts)


# class Observation(nn.Module):
#     d_state: int
#     d_obs: int
#     initializer: nn.initializers.Initializer = nn.initializers.normal(stddev=1.)
#
#     def setup(self):
#         self.seq = nn.Sequential([
#             nn.Dense(self.d_obs, kernel_init=self.initializer, use_bias=self.use_bias)
#         ])
#
#     def __call__(self, state):
#         return self.obs_matrix(state)
#
#     def observation_space(self, params):
#         return spaces.Box(-1, 1, (self.d_obs,), dtype=jnp.float32)


class Reward(nn.Module):
    d_state: int
    initializer: nn.initializers.Initializer = nn.initializers.normal(stddev=1.)
    dist_thresh: float = 3e-1

    def setup(self):
        self.state_goal = self.param('state_goal', self.initializer, (self.d_state,))

    def __call__(self, state):
        dist = jnp.linalg.norm(state - self.state_goal)
        rew = (dist < self.dist_thresh).astype(jnp.float32)
        return rew
