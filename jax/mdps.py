import jax
import jax.numpy as jnp
from flax import linen as nn
from einops import rearrange, repeat
import math


class LinearSyntheticMDP(nn.Module):
    n_actions: int
    d_state: int
    d_obs: int

    def setup(self):
        self.transition_fn = nn.Dense(features=self.n_actions * self.d_state, use_bias=False)
        self.obs_fn = nn.Dense(features=self.d_obs, use_bias=False)
        self.reward_fn = nn.Dense(features=1, use_bias=False)

    def __call__(self, state, action):
        state = self.transition_fn(state)
        state = rearrange(state, 'b (a d) -> b a d', a=self.n_actions)
        state = state[:, 0, :]
        obs = self.obs_fn(state)
        reward = self.reward_fn(state)
        return state, (obs, reward)

def main():
    rng = jax.random.PRNGKey(0)

    mdp = LinearSyntheticMDP(n_actions=5, d_state=8, d_obs=32)
    state = jnp.zeros((1, 8))
    action = jnp.zeros((1, ), dtype=jnp.int32)

    params = mdp.init(rng, state, action)

    print(jax.tree_map(lambda x: x.shape, params))


def main():
    rng = jax.random.PRNGKey(0)

if __name__ == '__main__':
    main()
