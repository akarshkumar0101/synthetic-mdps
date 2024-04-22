from functools import partial

import distrax
import jax
import jax.numpy as jnp
import optax
from einops import rearrange
from flax.training.train_state import TrainState
from jax.random import split

"""
RCBC is return conditioned behavior cloning.
RCBC assumes fixed episode lengths.
Resets the environment and agent every rollout.
Samples env_params = jax.vmap(env.sample_params)(split(_rng, n_envs)) every rollout.
"""


def make_rcbc_funcs(agent_collect, agent_dt, env,
                    n_envs=4, n_steps=128, n_updates=16, n_envs_batch=1, lr=2.5e-4, clip_grad_norm=0.5,
                    ent_coef=0.01, gamma=0.99):
    def rollout_step(carry, _):
        rng, agent_params, env_params, agent_state, obs, env_state = carry
        # agent
        rng, _rng = split(rng)
        forward_recurrent = partial(agent_collect.apply, method=agent_collect.forward_recurrent)
        agent_state_n, (logits, val) = jax.vmap(forward_recurrent, in_axes=(None, 0, 0))(agent_params, agent_state, obs)
        pi = distrax.Categorical(logits=logits)
        act = pi.sample(seed=_rng)
        log_prob = pi.log_prob(act)
        # env
        rng, _rng = split(rng)
        obs_n, env_state_n, rew, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, 0))(split(_rng, n_envs),
                                                                                       env_state, act, env_params)
        carry = rng, agent_params, env_params, agent_state_n, obs_n, env_state_n
        trans = dict(obs=obs, act=act, rew=rew, done=done, info=info, logits=logits, log_prob=log_prob, val=val,
                     env_state=env_state)
        return carry, trans

    def loss_fn(agent_params, batch):
        forward_parallel = partial(agent_dt.apply, method=agent_dt.forward_parallel)
        rtg_obs = dict(rtg=batch['rtg'], **batch['obs'])
        logits, val = jax.vmap(forward_parallel, in_axes=(None, 0))(agent_params, rtg_obs)
        pi = distrax.Categorical(logits=logits)
        loss_actor = optax.softmax_cross_entropy_with_integer_labels(logits, batch['act'])
        entropy = pi.entropy().mean()
        loss = 1.0 * loss_actor.mean() + - ent_coef * entropy
        return loss, (loss_actor, entropy)

    def update_batch(carry, _):
        rng, train_state, buffer = carry

        rng, _rng = split(rng)
        idx_env = jax.random.permutation(_rng, n_envs)[:n_envs_batch]
        batch = jax.tree_util.tree_map(lambda x: x[:, idx_env], buffer)
        batch = jax.tree_util.tree_map(lambda x: rearrange(x, 't n ... -> n t ...'), batch)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        loss, grads = grad_fn(train_state.params, batch)
        train_state = train_state.apply_gradients(grads=grads)
        carry = rng, train_state, buffer
        return carry, loss

    def eval_step(carry, _):
        rng, agent_params, train_state, env_params, agent_state, obs, env_state = carry

        # resetting every rollout
        rng, _rng = split(rng)
        env_params = jax.vmap(env.sample_params)(split(_rng, n_envs))
        rng, _rng = split(rng)
        agent_state = jax.vmap(agent_collect.get_init_state)(split(_rng, n_envs))
        rng, _rng = split(rng)
        obs, env_state = jax.vmap(env.reset)(split(_rng, n_envs), env_params)

        carry = rng, agent_params, env_params, agent_state, obs, env_state
        carry, buffer = jax.lax.scan(rollout_step, carry, None, n_steps)
        rng, agent_params, env_params, agent_state, obs, env_state = carry

        carry = rng, agent_params, train_state, env_params, agent_state, obs, env_state
        return carry, buffer

    def dt_step(carry, _):
        carry, buffer = eval_step(carry, None)
        rng, agent_params, train_state, env_params, agent_state, obs, env_state = carry

        buffer['rtg'] = jnp.cumsum(buffer['rew'][::-1], axis=0)[::-1]  # reverse cumsum

        carry = rng, train_state, buffer
        carry, losses = jax.lax.scan(update_batch, carry, None, n_updates)
        rng, train_state, buffer = carry

        carry = rng, agent_params, train_state, env_params, agent_state, obs, env_state
        return carry, (buffer, losses)

    def init_agent_env(rng):
        rng, _rng = split(rng)
        env_params = jax.vmap(env.sample_params)(split(_rng, n_envs))
        rng, _rng = split(rng)
        obs, env_state = jax.vmap(env.reset, in_axes=(0, 0))(split(_rng, n_envs), env_params)

        rng, _rng = split(rng)
        agent_state = jax.vmap(agent_dt.get_init_state)(split(_rng, n_envs))
        agent_state0, obs0 = jax.tree_map(lambda x: x[0], (agent_state, obs))
        rng, _rng = split(rng)
        agent_params_dt = agent_dt.init(_rng, agent_state0, dict(rtg=jnp.zeros(()), **obs0), method=agent_dt.forward_recurrent)

        rng, _rng = split(rng)
        agent_state = jax.vmap(agent_collect.get_init_state)(split(_rng, n_envs))
        agent_state0, obs0 = jax.tree_map(lambda x: x[0], (agent_state, obs))
        rng, _rng = split(rng)
        agent_params = agent_collect.init(_rng, agent_state0, obs0, method=agent_collect.forward_recurrent)

        tx = optax.chain(optax.clip_by_global_norm(clip_grad_norm), optax.adam(lr, eps=1e-5))
        train_state = TrainState.create(apply_fn=agent_dt.apply, params=agent_params_dt, tx=tx)
        return rng, agent_params, train_state, env_params, agent_state, obs, env_state

    return init_agent_env, dt_step
