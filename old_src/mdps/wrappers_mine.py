from functools import partial
from typing import Tuple, Union, Optional

import chex
import gymnax.environments.spaces
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments.environment import EnvState, EnvParams
from jax.random import split

from agents import RandomAgent
from algos.ppo_dr import PPO


# class DictObsEnvironment(GymnaxWrapper):
#     @partial(jax.jit, static_argnums=(0,))
#     def step(self,
#              key: chex.PRNGKey,
#              state: EnvState,
#              action: Union[int, float],
#              params: Optional[EnvParams] = None,
#              ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
#         """Performs step transitions in the environment."""
#         # Use default env parameters if no others specified
#         if params is None:
#             params = self.default_params
#         key, key_reset = jax.random.split(key)
#         obs_st, state_st, reward, done, info = self.step_env(
#             key, state, action, params
#         )
#         obs_re, state_re = self.reset_env(key_reset, params)
#         # Auto-reset environment based on termination
#         state = jax.tree_map(
#             lambda x, y: jax.lax.select(done, x, y), state_re, state_st
#         )
#         obs = jax.tree_map(
#             lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st
#         )
#         return obs, state, reward, done, info


class MyGymnaxWrapper(object):
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @partial(jax.jit, static_argnums=(0,))
    def step(
            self,
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
        # obs = jax.lax.select(done, obs_re, obs_st)
        obs = jax.tree_map(
            lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st
        )
        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(
            self, key: chex.PRNGKey, params: Optional[EnvParams] = None
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        obs, state = self.reset_env(key, params)
        return obs, state

    def step_env(
            self,
            key: chex.PRNGKey,
            state: EnvState,
            action: Union[int, float],
            params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Environment-specific step transition."""
        return self._env.step_env(key, state, action, params)

    def reset_env(
            self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Environment-specific reset."""
        return self._env.reset_env(key, params)


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


class TimeLimit(MyGymnaxWrapper):
    def __init__(self, env, n_steps_max):
        super().__init__(env)
        self.n_steps_max = n_steps_max

    def reset_env(self, key, params=None):
        obs, env_state = self._env.reset_env(key, params)
        state = dict(env_state=env_state, time=0)
        return obs, state

    def step_env(self, key, state, action, params=None):
        env_state, time = state['env_state'], state['time']
        obs, env_state, reward, done, info = self._env.step_env(key, env_state, action, params)
        state = dict(env_state=env_state, time=time + 1)
        done = jnp.logical_or(done, state['time'] >= self.n_steps_max)
        return obs, state, reward, done, info


class FlattenObservationWrapper(MyGymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation_space(self, params):
        obs_space = self._env.observation_space(params)
        assert isinstance(obs_space, gymnax.environments.spaces.Box)
        return gymnax.environments.spaces.Box(-1, 1, (np.prod(obs_space.shape),), obs_space.dtype)

    def reset_env(self, rng, params):
        obs, state = self._env.reset_env(rng, params)
        obs = obs.flatten()
        return obs, state

    def step_env(self, rng, state, action, params):
        obs, state, reward, done, info = self._env.step_env(rng, state, action, params)
        obs = obs.flatten()
        return obs, state, reward, done, info


class RandomlyProjectObservation(MyGymnaxWrapper):
    def __init__(self, env, d_out):
        super().__init__(env)
        self.d_out = d_out

    def sample_params(self, rng):
        rng, _rng = split(rng)
        env_params = self._env.sample_params(_rng)
        obs_space = self._env.observation_space(env_params)
        assert isinstance(obs_space, gymnax.environments.spaces.Box)
        assert len(obs_space.shape) == 1  # only works for flat observations
        d_in = obs_space.shape[0]
        rng, _rng = split(rng)
        A = jax.random.normal(_rng, (self.d_out, d_in))
        return dict(env_params=env_params, A=A)

    def observation_space(self, params):
        obs_space = self._env.observation_space(params['env_params'])
        assert isinstance(obs_space, gymnax.environments.spaces.Box)
        return gymnax.environments.spaces.Box(-1, 1, (self.d_out,), obs_space.dtype)

    def reset_env(self, rng, params):
        obs, state = self._env.reset_env(rng, params['env_params'])
        obs = params['A'] @ obs
        return obs, state

    def step_env(self, rng, state, action, params):
        obs, state, reward, done, info = self._env.step_env(rng, state, action, params['env_params'])
        obs = params['A'] @ obs
        return obs, state, reward, done, info


class DoneObsActRew(MyGymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset_env(self, rng, params=None):
        obs, state = self._env.reset_env(rng, params)
        obs = dict(done=1, obs=obs, act_p=0, rew_p=0.)
        return obs, state

    def step_env(self, rng, state, action, params=None):
        obs, state, rew, done, info = self._env.step_env(rng, state, action, params)
        obs = dict(done=0, obs=obs, act_p=action, rew_p=rew)
        return obs, state, rew, done, info

    def observation_space(self, params=None):
        done_space = gymnax.environments.spaces.Discrete(2)
        obs_space = self._env.observation_space(params)
        act_space = self._env.action_space(params)
        rew_space = gymnax.environments.spaces.Box(-1, 1, ())
        return gymnax.environments.spaces.Dict(dict(done=done_space, obs=obs_space, act_p=act_space, rew_p=rew_space))


class MetaRLWrapper(MyGymnaxWrapper):
    def __init__(self, env, n_trials, n_steps_trial):
        super().__init__(env)
        self.n_trials = n_trials
        self.n_steps_trial = n_steps_trial

    def reset_env(self, key, params=None):
        obs, _state = self._env.reset_env(key, params)
        state = dict(_state=_state, time=0)
        return obs, state

    def step_env(self, key, state, action, params=None):
        _state, time = state['_state'], state['time']
        done_trial = (time + 1) % self.n_steps_trial == 0
        done_all = (time + 1) % (self.n_steps_trial * self.n_trials) == 0

        obs, _state, rew, done, info = self._env.step_env(key, _state, action, params)
        obs_re, _state_re = self._env.reset_env(key, params)
        obs, _state = jax.tree_map(lambda x, y: jax.lax.select(done_trial, x, y), (obs_re, _state_re), (obs, _state))
        state = dict(_state=_state, time=time + 1)

        info['done_trial'] = done_trial
        done = done_all
        return obs, state, rew, done, info


def collect_random_agent(env, rng, n_envs, n_steps):
    rng, _rng = split(rng)
    env_params = jax.vmap(env.sample_params)(split(_rng, n_envs))
    rng, _rng = split(rng)
    obs, state = jax.vmap(env.reset)(split(_rng, n_envs), env_params)
    os, rs = [], []
    for t in range(n_steps):
        rng, _rng = split(rng)
        act = jax.random.randint(_rng, (n_envs,), 0, env.n_acts)
        rng, _rng = split(rng)
        obs, state, rew, done, info = jax.vmap(env.step)(split(_rng, n_envs), state, act, env_params)
        os.append(obs)
        rs.append(rew)
    os, rs = jnp.stack(os, axis=1), jnp.stack(rs, axis=1)
    return os, rs


class GaussianObsReward(MyGymnaxWrapper):
    def __init__(self, env, n_envs, n_steps):
        super().__init__(env)
        obs, rews = collect_random_agent(env, jax.random.PRNGKey(0), n_envs, n_steps)
        self.mean_obs, self.std_obs = obs.mean(axis=(0, 1)), obs.std(axis=(0, 1))
        self.mean_rew, self.std_rew = rews.mean(axis=(0, 1)), rews.std(axis=(0, 1))
        print(f'GaussianObsReward: mean_obs={self.mean_obs}, std_obs={self.std_obs}')
        print(f'GaussianObsReward: mean_rew={self.mean_rew}, std_rew={self.std_rew}')

    def reset_env(self, rng, params=None):
        obs, state = self._env.reset_env(rng, params)
        obs = (obs - self.mean_obs) / self.std_obs
        return obs, state

    def step_env(self, rng, state, action, params=None):
        obs, state, rew, done, info = self._env.step_env(rng, state, action, params)
        obs = (obs - self.mean_obs) / self.std_obs
        rew = (rew - self.mean_rew) / self.std_rew
        return obs, state, rew, done, info


class ObsNormRand(MyGymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)

        n_acts = env.action_space(None).n
        agent = RandomAgent(n_acts)
        ppo = PPO(agent, env, sample_env_params=env.sample_params, n_envs=1024, n_steps=1024)

        rng = jax.random.PRNGKey(0)
        carry = ppo.init_agent_env(rng)
        carry, buffer = ppo.eval_step(carry, None)
        obs = buffer['obs']
        self.mean = obs.mean(axis=(0, 1))
        self.std = obs.std(axis=(0, 1))
        print(buffer['rew'].mean())

    def reset_env(self, key, params):
        obs, state = self._env.reset_env(key, params)
        obs = (obs - self.mean) / self.std
        return obs, state

    def step_env(self, key, state, action, params):
        obs, state, rew, done, info = self._env.step_env(key, state, action, params)
        obs = (obs - self.mean) / self.std
        return obs, state, rew, done, info