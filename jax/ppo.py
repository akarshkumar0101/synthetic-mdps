import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from flax.training.train_state import TrainState

from einops import rearrange, repeat
import math

from functools import partial
from agent import Transformer
from mdps import LinearSyntheticMDP

import gymnax


def rollout(model, env, rng, agent_params, env_params, agent_state, env_state, oart, n_steps):
    """ Batch with jax.vmap """
    def rollout_step(carry, _):
        rng, agent_state, env_state, oart = carry
        obs, act_p, rew_p, time = oart
        rng, _rng_agent, _rng_env = jax.random.split(rng, 3)
        agent_state, (logits, val) = model.apply(agent_params, agent_state, (obs, act_p, rew_p, time),
                                                 method=model.forward_recurrent)
        act = jax.random.categorical(_rng_agent, logits=logits).astype(jnp.int32)
        obs_n, state_n, rew, done, info = env.step(_rng_env, env_state, act, env_params)
        oart = obs_n, act, rew, time + 1
        carry = rng, agent_state, env_state, oart
        return carry, dict(obs=obs, act=act, rew=rew, done=done,
                           time=time, logits=logits, val=val, act_p=act_p, rew_p=rew_p)
    carry = rng, agent_state, env_state, oart
    carry, buffer = jax.lax.scan(rollout_step, carry, jnp.arange(n_steps))
    return carry, buffer


def calc_gae(buffer, val_last, gamma=0.99, gae_lambda=0.95):
    """ Batch with jax.vmap """
    def gae_step(carry, transition):
        gae, next_value = carry
        done, value, reward = transition['done'], transition['val'], transition['rew']
        delta = reward + gamma * next_value * (1 - done) - value
        gae = (delta + gamma * gae_lambda * (1 - done) * gae)
        return (gae, value), gae
    carry = jnp.zeros_like(val_last), val_last
    _, adv = jax.lax.scan(gae_step, carry, buffer, reverse=True, unroll=16)
    ret = adv + buffer['val']
    return adv, ret


def main():
    config = dict(lr=1e-3, bs=3, n_envs=8, n_steps=128, n_acts=10, d_state=8, d_obs=32, d_embd=128,
                  n_updates=10, n_iters=20)
    clip_coef = 0.2
    vf_coef = 0.5
    ent_coef = 0.01
    rng = jax.random.PRNGKey(0)

    env, env_params = gymnax.make("CartPole-v1")
    env_params = jax.tree_map(lambda x: jnp.array([x for _ in range(config['n_envs'])]), env_params)

    rng, *_rng = jax.random.split(rng, 1 + config['n_envs'])
    obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(jnp.stack(_rng), env_params)
    act = jnp.zeros((config['n_envs'],), dtype=jnp.int32)
    rew = jnp.zeros((config['n_envs'],))
    time = jnp.zeros((config['n_envs'],), dtype=jnp.int32)
    oart = obs, act, rew, time

    model = Transformer(n_actions=env.action_space(env_params).n, n_steps=config['n_steps'],
                        n_layers=3, n_heads=4, d_embd=32)

    oart_bt = jax.tree_map(lambda x: repeat(x, 'b ... -> (b t) ...', t=config['n_steps']), oart)
    rng, _rng = jax.random.split(rng)
    agent_params = model.init(_rng, *oart_bt)
    agent_state = model.get_init_state(config['n_envs'])
    print('agent_params: ')
    print(jax.tree_map(lambda x: x.shape, agent_params))
    print('agent_state: ')
    print(jax.tree_map(lambda x: x.shape, agent_state))

    tx = optax.chain(optax.clip_by_global_norm(1.), optax.adam(config['lr'], eps=1e-5), )
    train_state = TrainState.create(apply_fn=partial(model.apply, method=model.forward_parallel),
                                    params=agent_params, tx=tx)
    runner_state = rng, train_state

    def ppo_step(runner_state, _):
        rng, train_state = runner_state
        rng, *_rng = jax.random.split(rng, 1 + config['n_envs'])
        _rng = jnp.stack(_rng)

        # def rollout(model, env, rng, agent_params, env_params, agent_state, env_state, oart, n_steps):
        batch_rollout = jax.vmap(rollout, in_axes=(None, None, 0, None, 0, 0, 0, 0, None))
        _, buffer = batch_rollout(model, env, _rng, agent_params, env_params, agent_state, env_state, oart, config['n_steps'])
        print('buffer: ')
        print(jax.tree_map(lambda x: x.shape, buffer))
        val_last = buffer['val'][..., 0]

        # def calc_gae(traj_batch, val_last, gamma=0.99, gae_lambda=0.95):
        batch_calc_gae = jax.vmap(calc_gae, in_axes=(0, 0, None, None))
        adv, ret = batch_calc_gae(buffer, val_last, 0.99, 0.95)
        buffer.update(adv=adv, ret=ret)

        def update_step(train_state, rng):
            def loss_fn(params, batch):
                logits, val = model.apply(params, batch['obs'], batch['act_p'], batch['rew_p'], batch['time'],
                                          method=model.forward_parallel)
                logits_old, val_old = batch['logits'], batch['val']
                act = batch['act']
                adv, ret = batch['adv'], batch['ret']
                log_prob_old = jax.nn.log_softmax(logits_old)[jnp.arange(config['n_steps']), act]
                log_prob = jax.nn.log_softmax(logits)[jnp.arange(config['n_steps']), act]

                value_pred_clipped = val_old + (val - val_old).clip(-clip_coef, clip_coef)
                value_losses = jnp.square(val - ret)
                value_losses_clipped = jnp.square(value_pred_clipped - ret)
                value_loss = (0.5 * jnp.maximum(value_losses, value_losses_clipped).mean())

                ratio = jnp.exp(log_prob - log_prob_old)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                loss_actor1 = ratio * adv
                loss_actor2 = jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef, ) * adv
                loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                loss_actor = loss_actor.mean()

                pi = jax.nn.softmax(logits)
                logpi = jax.nn.log_softmax(logits)
                entropy = -jnp.sum(pi * logpi, axis=-1).mean()

                loss = loss_actor + vf_coef * value_loss - ent_coef * entropy
                return loss, (value_loss, loss_actor, entropy)

            def batch_loss_fn(params, batch):
                loss, _ = jax.vmap(loss_fn, in_axes=(None, 0))(params, batch)
                return jnp.mean(loss, axis=0)

            idx = jax.random.permutation(rng, n_envs)
            batch = jax.tree_map(lambda x: x[idx], buffer)
            valgrad_fn = jax.value_and_grad(batch_loss_fn)
            loss, grads = valgrad_fn(train_state.params, batch)
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, None

        rng, *_rng = jax.random.split(rng, 1 + config['n_updates'])
        train_state, _ = jax.lax.scan(update_step, train_state, jnp.stack(_rng))


if __name__ == '__main__':
    main()

"""
----- WHAT THE BUFFER IS -----
observation: o0, o1, o2
    actions: a0, a1, a2
    rewards: r0, r1, r2
      times: t0, t1, t2

----- WHAT THE MODEL SEES -----
 prediction: a0, a1, a2
observation: o0, o1, o2
    actions: __, a0, a1, a2
    rewards: __, r0, r1, r2
      times: t0, t1, t2
"""
