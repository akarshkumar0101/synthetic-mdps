from functools import partial

import distrax
import jax
import jax.numpy as jnp
import optax
from einops import rearrange
from flax.training.train_state import TrainState
from jax.random import split


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


def make_ppo_funcs(agent, env,
                   n_envs=4, n_steps=128, n_updates=16, n_envs_batch=1, lr=2.5e-4, clip_grad_norm=0.5,
                   clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, gamma=0.99, gae_lambda=0.95):
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
        trans = dict(obs=obs, act=act, rew=rew, done=done, info=info, logits=logits, log_prob=log_prob, val=val,
                     env_state=env_state)
        return carry, trans

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

    def ppo_step(carry, _):
        rng, train_state, env_params, agent_state, obs, env_state = carry
        agent_params = train_state.params

        # resetting every rollout
        rng, _rng = split(rng)
        env_params = jax.vmap(env.sample_params)(split(_rng, n_envs))
        rng, _rng = split(rng)
        agent_state = jax.vmap(agent.get_init_state)(split(_rng, n_envs))
        rng, _rng = split(rng)
        obs, env_state = jax.vmap(env.reset)(split(_rng, n_envs), env_params)

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

    def eval_step(carry, _):
        rng, train_state, env_params, agent_state, obs, env_state = carry
        agent_params = train_state.params

        carry = rng, agent_params, env_params, agent_state, obs, env_state
        carry, buffer = jax.lax.scan(rollout_step, carry, None, n_steps)
        rng, agent_params, env_params, agent_state, obs, env_state = carry

        carry = rng, train_state, env_params, agent_state, obs, env_state
        return carry, buffer

    def init_agent_env(rng):
        rng, _rng = split(rng)
        env_params = jax.vmap(env.sample_params)(split(_rng, n_envs))
        rng, _rng = split(rng)
        agent_state = jax.vmap(agent.get_init_state)(split(_rng, n_envs))
        rng, _rng = split(rng)
        obs, env_state = jax.vmap(env.reset)(split(_rng, n_envs), env_params)

        agent_state0, obs0 = jax.tree_map(lambda x: x[0], (agent_state, obs))
        agent_params = agent.init(_rng, agent_state0, obs0, method=agent.forward_recurrent)

        tx = optax.chain(optax.clip_by_global_norm(clip_grad_norm), optax.adam(lr, eps=1e-5))
        train_state = TrainState.create(apply_fn=agent.apply, params=agent_params, tx=tx)
        return rng, train_state, env_params, agent_state, obs, env_state

    return init_agent_env, ppo_step, eval_step
