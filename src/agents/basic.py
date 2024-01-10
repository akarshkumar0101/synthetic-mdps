import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from .util import Agent


class BasicAgent(Agent):
    ObsEmbed: nn.Module
    n_acts: int

    activation: str = "tanh"

    def setup(self):
        self.obs_embed = self.ObsEmbed()
        activation = getattr(nn, self.activation)
        self.seq_main = nn.Sequential([
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            activation,
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            activation,
        ])
        self.actor = nn.Dense(self.n_acts, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def __call__(self, state, x):  # state.shape: (...), x.shape: (T, ...)
        state, x = self.obs_embed(state, x)
        x = self.seq_main(x)
        logits, val = self.actor(x), self.critic(x)  # (T, A) and (T, 1)
        return state, (logits, val[..., 0])

    def init_state(self, rng):
        return self.obs_embed.init_state(rng)


class BasicAgentSeparate(Agent):
    ObsEmbed: nn.Module
    n_acts: int

    activation: str = "tanh"

    def setup(self):
        self.obs_embed = self.ObsEmbed()
        activation = getattr(nn, self.activation)
        self.seq_pi = nn.Sequential([
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            activation,
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            activation,
            nn.Dense(self.n_acts, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        ])
        self.seq_critic = nn.Sequential([
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            activation,
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            activation,
            nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)),
        ])

    def __call__(self, state, x):  # state.shape: (...), x.shape: (T, ...)
        state, x = self.obs_embed(state, x)
        logits, val = self.seq_pi(x), self.seq_critic(x)  # (T, A) and (T, 1)
        return state, (logits, val[..., 0])

    def init_state(self, rng):
        return self.obs_embed.init_state(rng)


class RandomAgent(Agent):
    n_acts: int

    @nn.compact
    def __call__(self, state, x):  # state.shape: (...), x.shape: (T, ...)
        logits, val = jnp.zeros((x.shape[0], self.n_acts))
        val = jnp.zeros((x.shape[0], 1,))
        return state, (logits, val[..., 0])
