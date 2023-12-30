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

    _, adv = jax.lax.scan(calc_gae_step, (jnp.zeros_like(val_last), val_last), buffer, reverse=True)
    ret = adv + buffer['val']
    return adv, ret


class PPO:
    def __init__(self, agent, env, sample_env_params=None, *,
                 n_envs=4, n_steps=128, n_updates=16, n_envs_batch=1, lr=2.5e-4, clip_grad_norm=0.5,
                 clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, gamma=0.99, gae_lambda=0.95):
        self.agent, self.env = agent, env
        self.sample_env_params = self.env.sample_params if sample_env_params is None else sample_env_params
        self.config = dict(n_envs=n_envs, n_steps=n_steps, n_updates=n_updates, n_envs_batch=n_envs_batch,
                           lr=lr, clip_grad_norm=clip_grad_norm, clip_eps=clip_eps, vf_coef=vf_coef, ent_coef=ent_coef,
                           gamma=gamma, gae_lambda=gae_lambda)
        self.n_envs, self.n_steps = n_envs, n_steps
        self.n_updates, self.n_envs_batch = n_updates, n_envs_batch
        self.lr, self.clip_grad_norm, self.clip_eps = lr, clip_grad_norm, clip_eps
        self.vf_coef, self.ent_coef = vf_coef, ent_coef
        self.gamma, self.gae_lambda = gamma, gae_lambda

    def rollout_step(self, carry, _):
        rng, agent_params, env_params, agent_state, obs, env_state = carry
        # agent
        forward_recurrent = partial(self.agent.apply, method=self.agent.forward_recurrent)
        agent_state_n, (logits, val) = jax.vmap(forward_recurrent, in_axes=(None, 0, 0))(agent_params, agent_state, obs)
        pi = distrax.Categorical(logits=logits)
        rng, _rng = split(rng)
        act = pi.sample(seed=_rng)
        log_prob = pi.log_prob(act)
        # env
        rng, _rng = split(rng)
        obs_n, env_state_n, rew, done, info = jax.vmap(self.env.step)(split(_rng, self.n_envs), env_state, act,
                                                                      env_params)

        rng, _rng = split(rng)
        env_params_new = jax.vmap(self.sample_env_params)(split(_rng, self.n_envs))
        env_params = jax.tree_map(lambda x, y: jax.lax.select(done, x, y), env_params_new, env_params)

        carry = rng, agent_params, env_params, agent_state_n, obs_n, env_state_n
        trans = dict(obs=obs, act=act, rew=rew, done=done, info=info,
                     logits=logits, log_prob=log_prob, val=val,
                     agent_state=agent_state, env_state=env_state)
        return carry, trans

    def loss_fn(self, agent_params, batch):
        forward_parallel = partial(self.agent.apply, method=self.agent.forward_parallel)
        agent_state, obs = jax.tree_map(lambda x: x[:, 0], batch['agent_state']), batch['obs']
        _, (logits, val) = jax.vmap(forward_parallel, in_axes=(None, 0, 0))(agent_params, agent_state, obs)
        pi = distrax.Categorical(logits=logits)
        log_prob = pi.log_prob(batch['act'])

        # value loss
        value_pred_clipped = batch['val'] + (val - batch['val']).clip(-self.clip_eps, self.clip_eps)
        value_losses = jnp.square(val - batch['ret'])
        value_losses_clipped = jnp.square(value_pred_clipped - batch['ret'])
        value_loss = (0.5 * jnp.maximum(value_losses, value_losses_clipped).mean())

        # policy loss
        ratio = jnp.exp(log_prob - batch['log_prob'])
        gae = batch['adv']
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        loss_actor1 = ratio * gae
        loss_actor2 = jnp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * gae
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.mean()
        entropy = pi.entropy().mean()

        loss = 1.0 * loss_actor + self.vf_coef * value_loss - self.ent_coef * entropy
        return loss, (value_loss, loss_actor, entropy)

    def update_batch(self, carry, _):
        rng, train_state, buffer = carry

        rng, _rng = split(rng)
        idx_env = jax.random.permutation(_rng, self.n_envs)[:self.n_envs_batch]
        batch = jax.tree_util.tree_map(lambda x: x[:, idx_env], buffer)
        batch = jax.tree_util.tree_map(lambda x: rearrange(x, 't n ... -> n t ...'), batch)

        grad_fn = jax.value_and_grad(self.loss_fn, has_aux=True)
        loss, grads = grad_fn(train_state.params, batch)
        train_state = train_state.apply_gradients(grads=grads)
        carry = rng, train_state, buffer
        return carry, loss

    def eval_step(self, carry, _):
        rng, train_state, env_params, agent_state, obs, env_state = carry
        agent_params = train_state.params

        carry = rng, agent_params, env_params, agent_state, obs, env_state
        carry, buffer = jax.lax.scan(self.rollout_step, carry, None, self.n_steps)
        rng, agent_params, env_params, agent_state, obs, env_state = carry

        carry = rng, train_state, env_params, agent_state, obs, env_state
        return carry, buffer

    def ppo_step(self, carry, _):
        carry, buffer = self.eval_step(carry, None)
        rng, train_state, env_params, agent_state, obs, env_state = carry

        val_last = buffer['val'][-1]
        buffer['adv'], buffer['ret'] = calc_gae(buffer, val_last, gamma=self.gamma, gae_lambda=self.gae_lambda)

        carry = rng, train_state, buffer
        carry, losses = jax.lax.scan(self.update_batch, carry, None, self.n_updates)
        rng, train_state, buffer = carry

        carry = rng, train_state, env_params, agent_state, obs, env_state
        return carry, buffer

    def init_agent_env(self, rng):
        rng, _rng = split(rng)
        env_params = jax.vmap(self.sample_env_params)(split(_rng, self.n_envs))
        rng, _rng = split(rng)
        agent_state = jax.vmap(self.agent.get_init_state)(split(_rng, self.n_envs))
        rng, _rng = split(rng)
        obs, env_state = jax.vmap(self.env.reset)(split(_rng, self.n_envs), env_params)

        agent_state0, obs0 = jax.tree_map(lambda x: x[0], (agent_state, obs))
        rng, _rng = split(rng)
        agent_params = self.agent.init(_rng, agent_state0, obs0, method=self.agent.forward_recurrent)

        tx = optax.chain(optax.clip_by_global_norm(self.clip_grad_norm), optax.adam(self.lr, eps=1e-5))
        train_state = TrainState.create(apply_fn=self.agent.apply, params=agent_params, tx=tx)
        return rng, train_state, env_params, agent_state, obs, env_state
