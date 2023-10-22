import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct
import flax.linen as nn
from einops import rearrange, repeat


class GridEnv(environment.Environment):
    def __init__(self, grid_len):
        super().__init__()
        self.grid_len = grid_len
        self.obs = jnp.zeros((grid_len, grid_len), dtype=jnp.float32) - 1.

    def sample_params(self, rng):
        pos_rew = jax.random.randint(rng, (2, ), 0, self.grid_len)
        params = dict(pos_rew=pos_rew)
        return params

    # @property
    # def default_params(self) -> EnvParams:
    #     return EnvParams()

    def step_env(self, rng, state, action, params):
        """Performs step transitions in the environment."""
        pos_rew = params['pos_rew']

        # state = jax.vmap(self.model_trans.apply, in_axes=(0, None, None))(params_trans, state, rng)[action]
        state = jax.vmap(self.model_trans.apply, in_axes=(0, None))(params_trans, state)[action]
        obs = self.model_obs.apply(params_obs, state)
        rew = self.model_rew.apply(params_rew, state)[..., 0]

        done = jnp.zeros(rew.shape, dtype=jnp.bool_)
        info = {}
        return obs, state, rew, done, info

    def reset_env(self, rng, params):
        """Performs resetting of environment."""
        state = jnp.zeros((2, ), dtype=jnp.int32)
        obs = self.get_obs(state)
        return obs, state

    def get_obs(self, state):
        obs = self.obs.copy()
        obs[state[0], state[1]] = 1.
        return obs

    # def get_obs

    @property
    def name(self) -> str:
        return "GridEnv"

    @property
    def num_actions(self) -> int:
        return 4

    def action_space(self, params):
        return spaces.Discrete(self.n_acts)

    def observation_space(self, params):
        return spaces.Box(-1, 1, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params):
        return spaces.Box(-1, 1, self.state_shape, dtype=jnp.float32)

