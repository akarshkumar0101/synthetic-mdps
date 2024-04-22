from typing import Sequence

import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax.linen.initializers import constant, orthogonal
from jax.random import split
from tqdm.auto import tqdm

from src.algos.ppo import make_ppo_funcs
from wrappers import LogWrapper, FlattenObservationWrapper


class ActorCritic(nn.Module):
    n_acts: Sequence[int]
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
        return None

    def forward_recurrent(self, state, obs):  # shape: (...)
        logits = self.seq_pi(obs)
        val = self.seq_critic(obs)
        return state, (logits, val[..., 0])

    def forward_parallel(self, obs):  # shape: (n_steps, ...)
        logits = self.seq_pi(obs)
        val = self.seq_critic(obs)
        return logits, val[..., 0]


def main():
    print('starting')
    rng = jax.random.PRNGKey(0)
    env, env_params = gymnax.make("CartPole-v1")
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    agent = ActorCritic(env.action_space(env_params).n, activation="tanh")

    init_agent_env, eval_step, ppo_step = make_ppo_funcs(agent, env)
    init_agent_env = jax.vmap(init_agent_env)
    eval_step = jax.jit(jax.vmap(eval_step, in_axes=(0, None)))
    ppo_step = jax.jit(jax.vmap(ppo_step, in_axes=(0, None)))

    carry = init_agent_env(split(rng, 32))
    rets = []
    pbar = tqdm(range(500))
    for _ in pbar:
        carry, buffer = ppo_step(carry, None)
        rets.append(buffer['info']['returned_episode_returns'])
        pbar.set_postfix(ret=rets[-1].mean())
    rets = jnp.stack(rets, axis=1)
    print(rets.shape)

    steps = jnp.arange(rets.shape[1]) * 128 * 4
    plt.plot(steps, jnp.mean(rets, axis=(0, 2, 3)), label='mean')
    plt.plot(steps, jnp.median(rets, axis=(0, 2, 3)), label='median')
    plt.plot(steps, jnp.mean(rets, axis=(2, 3)).T, c='gray', alpha=0.1)
    plt.legend()
    plt.ylabel('Return')
    plt.xlabel('Env Steps')
    plt.show()
    print('done')


def main_lin():
    print('starting')

    from src.mdps.wrappers_mine import DoneObsActRew
    from src.agents.linear_transformer import LinearTransformerAgent

    rng = jax.random.PRNGKey(0)
    env, env_params = gymnax.make("CartPole-v1")
    env = FlattenObservationWrapper(env)
    env = DoneObsActRew(env)
    env = LogWrapper(env)
    agent = LinearTransformerAgent(env.action_space(env_params).n, 128, 1, 4, 64)

    # TODO: linear transformer requires ppo_step to reset agent_state every rollout...
    init_agent_env, eval_step, ppo_step = make_ppo_funcs(agent, env)
    init_agent_env = jax.vmap(init_agent_env)
    eval_step = jax.jit(jax.vmap(eval_step, in_axes=(0, None)))
    ppo_step = jax.jit(jax.vmap(ppo_step, in_axes=(0, None)))

    carry = init_agent_env(split(rng, 4))
    rets = []
    pbar = tqdm(range(1000))
    for _ in pbar:
        carry, buffer = ppo_step(carry, None)
        rets.append(buffer['info']['returned_episode_returns'])
        pbar.set_postfix(ret=rets[-1].mean())
    rets = jnp.stack(rets, axis=1)
    print(rets.shape)

    steps = jnp.arange(rets.shape[1]) * 128 * 4
    plt.plot(steps, jnp.mean(rets, axis=(0, 2, 3)), label='mean')
    plt.plot(steps, jnp.median(rets, axis=(0, 2, 3)), label='median')
    plt.plot(steps, jnp.mean(rets, axis=(2, 3)).T, c='gray', alpha=0.1)
    plt.legend()
    plt.ylabel('Return')
    plt.xlabel('Env Steps')
    plt.show()
    print('done')


if __name__ == '__main__':
    main()
