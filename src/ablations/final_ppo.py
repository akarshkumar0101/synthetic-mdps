from functools import partial
from typing import Sequence

import distrax
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from einops import rearrange
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jax.random import split

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


def calc_gae(buffer, val_last, gamma=0.99, gae_lambda=0.95):
    def calc_gae_step(carry, trans):
        gae, val_n = carry
        done, val, rew = trans["done"], trans["val"], trans["rew"]
        delta = rew + gamma * val_n * (1 - done) - val
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        return (gae, val), gae

    carry = (jnp.zeros_like(val_last), val_last)
    _, adv = jax.lax.scan(calc_gae_step, carry, buffer, reverse=True)
    ret = adv + buffer['val']
    return adv, ret


def make_rollout_step_fn(agent, env, n_envs):
    def rollout_step(carry, _):
        rng, agent_params, env_params, agent_state, obs, env_state = carry
        # agent
        rng, _rng = split(rng)
        forward_recurrent = partial(agent.apply, method=agent.forward_recurrent)
        agent_state_n, (logits, val) = jax.vmap(forward_recurrent, in_axes=(None, 0, 0))(agent_params, agent_state, obs)
        pi = distrax.Categorical(logits=logits)
        act = pi.sample(seed=_rng)
        log_prob = pi.log_prob(act)
        # env
        rng, _rng = split(rng)
        obs_n, env_state_n, rew, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(split(_rng, n_envs),
                                                                                          env_state, act, env_params)
        carry = rng, agent_params, env_params, agent_state_n, obs_n, env_state_n
        trans = dict(obs=obs, act=act, rew=rew, done=done, info=info, logits=logits, log_prob=log_prob, val=val)
        return carry, trans

    return rollout_step


def make_update_batch_fn(agent, n_envs, n_envs_batch, clip_eps, vf_coef, ent_coef):
    def _loss_fn(agent_params, batch):
        forward_parallel = partial(agent.apply, method=agent.forward_parallel)
        logits, val = jax.vmap(forward_parallel, in_axes=(None, 0))(agent_params, batch['obs'])
        pi = distrax.Categorical(logits=logits)
        log_prob = pi.log_prob(batch['act'])

        # value loss
        value_pred_clipped = batch['val'] + (val - batch['val']).clip(-clip_eps, clip_eps)
        value_losses = jnp.square(val - batch['ret'])
        value_losses_clipped = jnp.square(value_pred_clipped - batch['ret'])
        value_loss = (0.5 * jnp.maximum(value_losses, value_losses_clipped).mean())

        # policy loss
        ratio = jnp.exp(log_prob - batch['log_prob'])
        gae = batch['adv']
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        loss_actor1 = ratio * gae
        loss_actor2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.mean()
        entropy = pi.entropy().mean()

        loss = 1.0 * loss_actor + vf_coef * value_loss - ent_coef * entropy
        return loss, (value_loss, loss_actor, entropy)

    def update_batch(carry, _):
        rng, train_state, buffer = carry

        rng, _rng = split(rng)
        idx_env = jax.random.permutation(_rng, n_envs)[:n_envs_batch]
        batch = jax.tree_util.tree_map(lambda x: x[:, idx_env], buffer)
        batch = jax.tree_util.tree_map(lambda x: rearrange(x, 't n ... -> n t ...'), batch)

        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        loss, grads = grad_fn(train_state.params, batch)
        train_state = train_state.apply_gradients(grads=grads)
        carry = rng, train_state, buffer
        return carry, loss

    return update_batch


def make_ppo_step_fn(agent, env, n_envs, n_steps, n_updates, n_envs_batch, clip_eps, vf_coef, ent_coef, gamma=0.99,
                     gae_lambda=0.95):
    rollout_step = make_rollout_step_fn(agent, env, n_envs)
    update_batch = make_update_batch_fn(agent, n_envs, n_envs_batch, clip_eps, vf_coef, ent_coef)

    def ppo_step(carry, _):
        rng, train_state, env_params, agent_state, obs, env_state = carry
        agent_params = train_state.params

        carry = rng, agent_params, env_params, agent_state, obs, env_state
        carry, buffer = jax.lax.scan(rollout_step, carry, None, n_steps)
        rng, agent_params, env_params, agent_state, obs, env_state = carry

        val_last = buffer['val'][-1]
        buffer['adv'], buffer['ret'] = calc_gae(buffer, val_last, gamma=gamma, gae_lambda=gae_lambda)

        carry = rng, train_state, buffer
        carry, losses = jax.lax.scan(update_batch, carry, None, n_updates)
        rng, train_state, buffer = carry

        carry = rng, train_state, env_params, agent_state, obs, env_state
        return carry, buffer['info']['returned_episode_returns']

    return ppo_step


def make_train():
    env, env_params = gymnax.make("CartPole-v1")
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    agent = ActorCritic(env.action_space(env_params).n, activation="tanh")
    n_envs = 4

    def train(rng):
        env_params = env.default_params
        rng, _rng = split(rng)
        agent_state = jax.vmap(agent.get_init_state)(split(_rng, n_envs))
        rng, _rng = split(rng)
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(split(rng, n_envs), env_params)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = agent.init(_rng, None, init_x, method=agent.forward_recurrent)
        tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(2.5e-4, eps=1e-5))
        train_state = TrainState.create(apply_fn=agent.apply, params=network_params, tx=tx, )

        ppo_step = make_ppo_step_fn(agent, env, n_envs, 128, 16, 1, 0.2, 0.5, 0.01, gamma=0.99, gae_lambda=0.95)

        carry = rng, train_state, env_params, agent_state, obs, env_state
        carry, rews = jax.lax.scan(ppo_step, carry, None, 1000)
        rng, train_state, env_params, agent_state, obs, env_state = carry
        return rews

    return train


def temp():
    print('hello')
    train = make_train()

    rng = jax.random.PRNGKey(0)
    rets = jax.vmap(train)(split(rng, 32))

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


if __name__ == "__main__":
    temp()
