import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces
import flax.linen as nn


class DiscreteTransition(nn.Module):
    n_states: int
    initializer: nn.initializers.Initializer = nn.initializers.normal(stddev=1.)

    def setup(self):
        self.trans_matrix = self.param('trans_matrix',
                                       self.initializer(), (self.n_states, self.n_states))

    def apply(self, state, rng):
        logits = self.trans_matrix[:, state]
        state_n = jax.random.categorical(rng, logits)
        return state_n


class DiscreteReward(nn.Module):
    n_states: int

    def setup(self):
        self.rew_matrix = self.param('rew_matrix', nn.initializers.normal(stddev=1.), (self.n_states,))

    def apply(self, state):
        return self.rew_matrix[state]
