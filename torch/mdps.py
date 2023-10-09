import torch
from torch import nn
from einops import rearrange, repeat
import math


class LinearSyntheticMDP(nn.Module):
    d_embd: int

    def setup(self):
        self.transition_fn = nn.Dense(features=self.d_embd)
        self.reward_fn = nn.Dense(features=1)
        self.state = self.variable('buffers', 'state', jnp.zeros, (20, 20))

    def __call__(self, state, action):
        state = self.transition_fn(state)
        reward = self.reward_fn(state)
        return state, reward


if __name__ == '__main__':
    rng = jax.random.PRNGKey(0)

    d_embd = 4
    mdp = LinearSyntheticMDP(d_embd=d_embd)
    state = jnp.zeros((1, d_embd))

    params = mdp.init(rng, state, jnp.zeros((1, 1)))

    print(jax.tree_map(lambda x: x.shape, params))

