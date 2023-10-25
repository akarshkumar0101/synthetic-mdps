import jax
import jax.numpy as jnp
from gymnax import EnvParams
from gymnax.environments import environment, spaces
from wrappers import GymnaxWrapper

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

    def action_space(self, params: EnvParams):
        return self._env.action_space(params)