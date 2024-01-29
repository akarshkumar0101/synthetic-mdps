import flax.linen as nn
import jax
import jax.numpy as jnp
from gymnax.environments import spaces


def eye(key, shape, dtype):
    return jnp.eye(*shape, dtype=dtype)


class DiscreteInit(nn.Module):
    n_states: int
    initializer: nn.initializers.Initializer = nn.initializers.zeros_init()

    def setup(self):
        self.logits = self.param('logits', self.initializer, (self.n_states,))

    def __call__(self, rng):
        return jax.random.categorical(rng, self.logits)


class DiscreteTransition(nn.Module):
    n_states: int
    initializer: nn.initializers.Initializer = nn.initializers.normal(stddev=1.)

    def setup(self):
        self.trans_matrix = self.param('trans_matrix', self.initializer, (self.n_states, self.n_states))

    def __call__(self, state, rng):
        logits = self.trans_matrix[:, state]
        state_n = jax.random.categorical(rng, logits)
        return state_n


class DiscreteObs(nn.Module):
    n_states: int
    d_obs: int
    initializer: nn.initializers.Initializer = eye

    def setup(self):
        self.embed = nn.Embed(self.n_states, self.d_obs, embedding_init=self.initializer)

    def __call__(self, state):
        return self.embed(state)

    def observation_space(self, params):
        return spaces.Box(-1, 1, (self.d_obs,), dtype=jnp.float32)


class DiscreteReward(nn.Module):
    n_states: int
    initializer: nn.initializers.Initializer = nn.initializers.uniform(scale=1.)

    # def initializer(self, rng, shape, dtype):
    #     return jax.random.uniform(rng, shape, dtype=dtype)

    def setup(self):
        self.rew_matrix = self.param('rew_matrix', self.initializer, (self.n_states,))

    def __call__(self, state):
        return self.rew_matrix[state]
