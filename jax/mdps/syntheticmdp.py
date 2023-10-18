import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct
import flax.linen as nn
from einops import rearrange, repeat


class SyntheticMDP(environment.Environment):
    def __init__(self, d_state, d_obs, n_acts, model_trans, model_obs, model_rew):
        super().__init__()
        self.state_shape = (d_state,)
        self.obs_shape = (d_obs,)
        self.n_acts = n_acts

        self.model_trans = model_trans  # state to state
        self.model_obs = model_obs  # state to obs
        self.model_rew = model_rew  # state to R^1

    def sample_params(self, rng):
        rng, _rng_trans, _rng_obs, _rng_rew = jax.random.split(rng, 4)
        state = jnp.zeros(self.state_shape)
        action = jnp.zeros((), dtype=jnp.int32)
        params_trans = self.model_trans.init(_rng_trans, state, action, rng)
        params_obs = self.model_obs.init(_rng_obs, state)
        params_rew = self.model_rew.init(_rng_rew, state)
        return params_trans, params_obs, params_rew

    # @property
    # def default_params(self) -> EnvParams:
    #     return EnvParams()

    def step_env(self, rng, state, action, params):
        """Performs step transitions in the environment."""
        params_trans, params_obs, params_rew = params

        state = self.model_trans.apply(params_trans, state, action, rng)
        obs = self.model_obs.apply(params_obs, state)
        rew = self.model_rew.apply(params_rew, state)[..., 0]

        done = jnp.zeros(rew.shape, dtype=jnp.bool_)
        info = {}
        return obs, state, rew, done, info

    def reset_env(self, rng, params):
        """Performs resetting of environment."""
        params_trans, params_obs, params_rew = params

        state = jax.random.normal(rng, self.state_shape)
        state = state / jnp.linalg.norm(state)
        obs = self.model_obs.apply(params_obs, state)
        return obs, state

    @property
    def name(self) -> str:
        """Environment name."""
        return "SyntheticMDP"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.n_acts

    def action_space(self, params):
        """Action space of the environment."""
        return spaces.Discrete(self.n_acts)

    def observation_space(self, params):
        """Observation space of the environment."""
        return spaces.Box(-1, 1, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params):
        """State space of the environment."""
        return spaces.Box(-1, 1, self.state_shape, dtype=jnp.float32)


class MultiActionFunction(nn.Module):
    model_class: nn.Module
    n_acts: int

    def setup(self, ):
        batch_model_class = nn.vmap(self.model_class, in_axes=0, out_axes=0,
                                    variable_axes={'params': 0}, split_rngs={'params': True})
        self.model = batch_model_class()

    def __call__(self, state, action, rng):
        state = repeat(state, '... -> a ...', a=self.n_acts)
        state = self.model(state)[action]
        return state / jnp.linalg.norm(state)


def main():
    rng = jax.random.PRNGKey(0)
    d_state = 10
    d_obs = 32
    n_acts = 4

    from functools import partial

    #
    # state = jnp.zeros((d_state,))
    # action = jnp.zeros((), dtype=jnp.int32)
    # params = model_trans.init(rng, state, action)
    # print(jax.tree_map(lambda x: x.shape, params))

    # Dense = partial(nn.Dense, features=d_state, use_bias=False)
    # BatchDense = nn.vmap(Dense, in_axes=0, out_axes=0, variable_axes={'params': 0}, split_rngs={'params': True})
    # model = BatchDense()
    # # model = nn.Dense(features=d_state, use_bias=False)
    #
    # rng, _rng = jax.random.split(rng)
    # state = jnp.zeros((4, d_state, ))
    # params = model.init(_rng, state)
    # print(jax.tree_map(lambda x: x.shape, params))

    # Dense = partial(nn.Dense, features=d_state, use_bias=False)
    # model_trans = MultiActionFunction(Dense, n_acts)
    # state = jnp.zeros((d_state,))
    # action = jnp.zeros((), dtype=jnp.int32)
    # params = model_trans.init(rng, state, action)
    # print(jax.tree_map(lambda x: x.shape, params))

    # model_trans = nn.Sequential([nn.Dense(features=d_state*n_acts, use_bias=False),
    #                              split_multiple_states, lambda x: x / jnp.linalg.norm(x)])
    # model_trans = TransitionModel(features=d_state * n_acts, use_bias=False)
    # model_trans = MultiActionFunction(partial(nn.Dense, features=d_state), n_acts)
    # model_obs = nn.Dense(features=d_obs, use_bias=False)
    # model_rew = nn.Dense(features=1, use_bias=False)
    #
    # mdp = SyntheticMDP(d_state, d_obs, n_acts, model_trans, model_obs, model_rew)
    #
    # rng, _rng = jax.random.split(rng)
    # env_params = mdp.sample_params(_rng)
    # print(jax.tree_map(lambda x: x.shape, env_params))
    #
    # rng, _rng = jax.random.split(rng)
    # obs, env_state = mdp.reset_env(_rng, env_params)
    # for i in range(10):
    #     rng, _rng = jax.random.split(rng)
    #     action = mdp.action_space(env_params).sample(_rng)
    #     print('env_state: ')
    #     print(jax.tree_map(lambda x: x.shape, env_state))
    #     print('action: ')
    #     print(jax.tree_map(lambda x: x.shape, action))
    #     rng, _rng = jax.random.split(rng)
    #     obs, env_state, rew, done, info = mdp.step(_rng, env_state, action, env_params)



    model_trans = nn.Sequential([nn.Dense(features=d_state*n_acts, use_bias=False),
                                 split_multiple_states, lambda x: x / jnp.linalg.norm(x)])


if __name__ == '__main__':
    main()
