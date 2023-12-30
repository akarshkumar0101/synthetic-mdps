from typing import Sequence

import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal
from jax.random import split
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from algos.ppo_dr import PPO
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

    def forward_recurrent(self, state, obs):  # state.shape: (...); obs.shape:(...)
        logits = self.seq_pi(obs)
        val = self.seq_critic(obs)
        return state, (logits, val[..., 0])

    def forward_parallel(self, state, obs):  # state.shape: (...); obs.shape:(T, ...)
        logits = self.seq_pi(obs)
        val = self.seq_critic(obs)
        return state, (logits, val[..., 0])


def main():
    print('starting')
    rng = jax.random.PRNGKey(0)
    env, env_params = gymnax.make("CartPole-v1")
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    agent = ActorCritic(env.action_space(env_params).n, activation="tanh")

    ppo = PPO(agent, env, sample_env_params=lambda rng: env.default_params)
    init_agent_env = jax.vmap(ppo.init_agent_env)
    eval_step = jax.jit(jax.vmap(ppo.eval_step, in_axes=(0, None)))
    ppo_step = jax.jit(jax.vmap(ppo.ppo_step, in_axes=(0, None)))

    rng, _rng = split(rng)
    carry = init_agent_env(split(_rng, 32))
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

def main_hyper():
    print('starting')
    rng = jax.random.PRNGKey(0)
    env, env_params = gymnax.make("CartPole-v1")
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    agent = ActorCritic(env.action_space(env_params).n, activation="tanh")

    def run_exp(lr):
        rng = jax.random.PRNGKey(0)
        ppo = PPO(agent, env, sample_env_params=lambda rng: env.default_params, lr=lr)
        init_agent_env = jax.vmap(ppo.init_agent_env)
        eval_step = jax.jit(jax.vmap(ppo.eval_step, in_axes=(0, None)))
        ppo_step = jax.jit(jax.vmap(ppo.ppo_step, in_axes=(0, None)))

        rng, _rng = split(rng)
        carry = init_agent_env(split(_rng, 16))
        rets = []
        for _ in range(100):
            carry, buffer = ppo_step(carry, None)
            rets.append(buffer['info']['returned_episode_returns'])
            # pbar.set_postfix(ret=rets[-1].mean())
        rets = jnp.stack(rets, axis=1)
        return rets

    lrs = jnp.logspace(-4, 0, 8)
    rets = jax.vmap(run_exp)(lrs)

    plt.plot(lrs, rets.mean(axis=(1, 3, 4))[:, -1])
    plt.xscale('log')
    plt.show()


if __name__ == '__main__':
    main_hyper()
