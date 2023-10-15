import jax
import jax.numpy as jnp
from flax import linen as nn
from einops import rearrange

class LinearSyntheticMDP(nn.Module):
    n_actions: int
    d_state: int
    d_obs: int

    def setup(self):
        self.transition_fn = nn.Dense(features=self.n_actions * self.d_state, use_bias=False)
        self.obs_fn = nn.Dense(features=self.d_obs, use_bias=False)
        self.reward_fn = nn.Dense(features=1, use_bias=False)

    def __call__(self, state, action):
        state = self.get_transition(state, action)
        obs, rew, done = self.get_obs_rew_done(state)
        return state, obs, rew, done

    def get_transition(self, state, action):
        state = self.transition_fn(state)
        state = rearrange(state, '(a d) -> a d', a=self.n_actions)
        state = state[0, :]
        return state

    def get_obs_rew_done(self, state):
        obs = self.obs_fn(state)
        rew = self.reward_fn(state)[..., 0]
        return obs, rew, jnp.zeros_like(rew)
