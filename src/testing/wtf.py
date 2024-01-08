import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from einop import einop
from einops import rearrange
from flax.linen.initializers import constant, orthogonal
from jax.random import split
from tqdm.auto import tqdm

from agents.util import MinAtarObsEmbed
from algos.ppo_dr import PPO
from wrappers import LogWrapper


class ActorCritic(nn.Module):
    n_acts: int
    activation: nn.tanh
    ObsEmbed: nn.Module

    def setup(self):
        self.obs_embed = self.ObsEmbed(d_embd=64)
        self.seq_pi = nn.Sequential([
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            self.activation,
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            self.activation,
            nn.Dense(self.n_acts, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        ])
        self.seq_critic = nn.Sequential([
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            self.activation,
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            self.activation,
            nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)),
        ])

    def init_state(self, rng):
        return self.obs_embed.init_state(rng)

    def __call__(self, state, x):  # state.shape: (...); obs.shape:(T, ...)
        state, x = self.obs_embed(state, x)
        logits = self.seq_pi(x)
        val = self.seq_critic(x)
        return state, (logits, val[..., 0])

    def forward_parallel(self, state, x):  # state.shape: (...), x.shape: (T, ...)
        return self(state, x)

    def forward_recurrent(self, state, x):  # state.shape: (...), x.shape: (...)
        x = jax.tree_map(lambda x: rearrange(x, '... -> 1 ...'), x)
        state, x = self(state, x)
        x = jax.tree_map(lambda x: rearrange(x, '1 ... -> ...'), x)
        return state, x


def main():
    rng = jax.random.PRNGKey(0)
    env, env_params = gymnax.make("Breakout-MinAtar")
    n_acts = env.action_space(env_params).n
    env = LogWrapper(env)
    agent = ActorCritic(n_acts, activation=nn.tanh, ObsEmbed=MinAtarObsEmbed)

    ppo = PPO(agent, env, sample_env_params=lambda rng: env.default_params)
    init_agent_env = jax.vmap(ppo.init_agent_env)
    eval_step = jax.jit(jax.vmap(ppo.eval_step, in_axes=(0, None)))
    ppo_step = jax.jit(jax.vmap(ppo.ppo_step, in_axes=(0, None)))

    rng, _rng = split(rng)
    carry = init_agent_env(split(_rng, 4))
    rets = []
    pbar = tqdm(range(3000))
    for _ in pbar:
        carry, buffer = ppo_step(carry, None)
        rets.append(buffer['info']['returned_episode_returns'])
        pbar.set_postfix(ret=rets[-1].mean())
    rets = jnp.stack(rets, axis=1)
    print(rets.shape)

    steps = jnp.arange(rets.shape[1]) * 128 * 4
    plt.plot(steps, einop(rets, 's n t e -> n', reduction='mean'), label='mean')
    plt.plot(steps, einop(rets, 's n t e -> n', reduction=jnp.median), label='median')
    plt.plot(steps, einop(rets, 's n t e -> n s', reduction='mean'), c='gray', alpha=0.1)
    plt.legend()
    plt.ylabel('Return')
    plt.xlabel('Env Steps')
    plt.show()
    print('done')


if __name__ == '__main__':
    main()
