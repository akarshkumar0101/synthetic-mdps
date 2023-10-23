import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal


class BasicAgent(nn.Module):
    n_acts: int
    activation: str = "tanh"

    def setup(self):
        activation = nn.relu if self.activation == "relu" else nn.tanh
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

    def get_init_state(self, rng):
        return jnp.zeros(())

    def forward_recurrent(self, state, oar):  # shape: (...)
        obs, act_p, rew_p = oar
        logits = self.seq_pi(obs)
        val = self.seq_critic(obs)
        # pi = distrax.Categorical(logits=logits)
        # return state, (pi, val[..., 0])
        return state, (logits, val[..., 0])

    def forward_parallel(self, obs, act_p, rew_p):  # shape: (n_steps, ...)
        logits = self.seq_pi(obs)
        val = self.seq_critic(obs)
        # pi = distrax.Categorical(logits=logits)
        return logits, val[..., 0]
