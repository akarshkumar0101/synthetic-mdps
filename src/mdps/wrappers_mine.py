from functools import partial
from typing import Tuple, Union, Optional

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments.environment import EnvState, EnvParams

from .wrappers import GymnaxWrapper

"""
This class doesn't work because Gymnax wrappers
call the underlying env step where the done is already implemented below.
Changing done after calling step doesn't do anything (won't auto reset env).
"""

# class TimeLimit(GymnaxWrapper):
#     def __init__(self, env, max_steps):
#         super().__init__(env)
#         self.max_steps = max_steps
#
#     def reset(self, key, params=None):
#         obs, env_state = self._env.reset(key, params)
#         state = dict(env_state=env_state, time=jnp.array(0, dtype=jnp.int32))
#         return obs, state
#
#     def step(self, key, state, action, params=None):
#         env_state, time = state['env_state'], state['time']
#         obs, env_state, reward, done, info = self._env.step(key, env_state, action, params)
#         state = dict(env_state=env_state, time=time + 1)
#         done = state['time'] >= self.max_steps
#         return obs, state, reward, done, info

"""
This class will work because subclassing Environment instead of Wrapper
gives us our own reset and step methods which handle auto-resetting.
Now we only need to implement reset_env and step_env.
"""


class TimeLimit(environment.Environment):
    def __init__(self, env, max_steps):
        self._env = env
        self.max_steps = max_steps

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset_env(self, key, params=None):
        obs, env_state = self._env.reset_env(key, params)
        state = dict(env_state=env_state, time=jnp.array(0, dtype=jnp.int32))
        return obs, state

    def step_env(self, key, state, action, params=None):
        env_state, time = state['env_state'], state['time']
        obs, env_state, reward, done, info = self._env.step_env(key, env_state, action, params)
        state = dict(env_state=env_state, time=time + 1)
        done = jnp.logical_or(done, state['time'] >= self.max_steps)
        return obs, state, reward, done, info

    def action_space(self, params):
        return self._env.action_space(params)

    def observation_space(self, params):
        return self._env.observation_space(params)


class RandomlyProjectObservation(GymnaxWrapper):
    def __init__(self, env, d_obs):
        super().__init__(env)
        self.d_obs = d_obs

    def sample_params(self, rng):
        rng, _rng = jax.random.split(rng)
        env_params = self._env.sample_params(rng)
        d_obs = self._env.observation_space(env_params).shape[0]
        A = jax.random.normal(_rng, (self.d_obs, d_obs))
        return {'env_params': env_params, 'A': A}

    def reset(self, key, params):
        obs, state = self._env.reset(key, params['env_params'])
        obs = params['A'] @ obs
        return obs, state

    def step(self, key, state, action, params):
        obs, state, reward, done, info = self._env.step(key, state, action, params['env_params'])
        obs = params['A'] @ obs
        return obs, state, reward, done, info


class NoReward(GymnaxWrapper):
    def step(self, key, state, action, params):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        reward = jnp.zeros_like(reward)
        return obs, state, reward, done, info


class RewardTransform(GymnaxWrapper):
    def __init__(self, env, reward_transform):
        super().__init__(env)
        self.reward_transform = reward_transform

    def step(self, key, state, action, params):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        reward = self.reward_transform(reward)
        return obs, state, reward, done, info


class DictObsEnvironment(object):

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.env.step = partial(self.step, self)

    def __getattr__(self, name):
        return getattr(self.env, name)

    @partial(jax.jit, static_argnums=(0,))
    def step(self,
             key: chex.PRNGKey,
             state: EnvState,
             action: Union[int, float],
             params: Optional[EnvParams] = None,
             ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(
            key, state, action, params
        )
        obs_re, state_re = self.reset_env(key_reset, params)
        # Auto-reset environment based on termination
        state = jax.tree_map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        obs = jax.tree_map(
            lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st
        )
        return obs, state, reward, done, info


class DoneObsActRew(DictObsEnvironment):
    def __init__(self, env):
        super().__init__(env)
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset_env(self, rng, params=None):
        obs, state = self._env.reset_env(rng, params)
        obs = dict(done=True, obs=obs, act_p=0, rew_p=0.)
        return obs, state

    def step_env(self, key, state, action, params=None):
        obs, state, rew, done, info = self._env.step_env(key, state, action, params)
        obs = dict(done=False, obs=obs, act_p=action, rew_p=rew)
        return obs, state, rew, done, info

    # def observation_space(self, params):
    #     obs_space = self._env.observation_space(params)
    #     act_space = self.action_space()
    #     gymnax.environments.spaces.Dict()
    #     return
