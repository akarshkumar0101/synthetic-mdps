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


def collect(rng, agent_state, env_state, forward_recurrent, transition_fn, obs_rew_done_fn, action, reward, time):
    def collect_step(carry, _):
        rng, agent_state_t, env_state_t, (act_tp, rew_tp, time_tp) = carry
        time_t = time_tp + 1
        obs_t, rew_t, done_t = obs_rew_done_fn(env_state_t)

        agent_state_tn, (logits_t, val_t) = forward_recurrent(agent_state_t, (obs_t, act_tp, rew_tp, time_t))
        rng, _rng = jax.random.split(rng)
        act_t = jax.random.categorical(_rng, logits=logits_t)
        env_state_tn = transition_fn(env_state_t, act_t)

        carry = rng, agent_state_tn, env_state_tn, (act_t, rew_t, time_t)
        return carry, dict(env_state=env_state_t, obs=obs_t, act_tp=act_tp, act=act_t, rew_tp=rew_tp, rew=rew_t,
                           time=time_t, logits=logits_t, val=val_t, done=done_t)

    carry = rng, agent_state, env_state, (action, reward, time)
    carry, buffer = jax.lax.scan(collect_step, carry, jnp.arange(30))
    rng, agent_state, env_state, (action, reward, time) = carry

    obs_last, rew_last, done_last = obs_rew_done_fn(env_state)
    agent_state, (_, val_last) = forward_recurrent(agent_state, (obs_last, action, reward, time))

    return agent_state, env_state, buffer, val_last


def calc_gae(traj_batch, val_last):
    def gae_step(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        done, value, reward = transition['done'], transition['val'], transition['rew']
        delta = reward + config["GAMMA"] * next_value * (1 - done) - value
        gae = (delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae)
        return (gae, value), gae

    _, advantages = jax.lax.scan(gae_step, (jnp.zeros_like(val_last), val_last), traj_batch, reverse=True, unroll=16)
    returns = advantages + traj_batch['val']
    return advantages, returns


def train(rng, config):
    n_actions = config['n_actions']
    d_state = config['d_state']
    d_obs = config['d_obs']
    d_embd = config['d_embd']
    n_steps = config['n_steps']
    bs = config['bs']

    net = Transformer(n_actions=n_actions, n_steps=n_steps, n_layers=3, n_heads=4, d_embd=d_embd)
    obs = jnp.zeros((bs, n_steps, d_obs))
    action = jnp.zeros((bs, n_steps), dtype=jnp.int32)
    reward = jnp.zeros((bs, n_steps))
    time = jnp.zeros((bs, n_steps), dtype=jnp.int32)

    rng, _rng = jax.random.split(rng)
    params = net.init(_rng, obs[0], action[0], reward[0], time[0])
    print(jax.tree_map(lambda x: x.shape, params))

    mdp = LinearSyntheticMDP(n_actions=n_actions, d_state=d_state, d_obs=d_obs)
    env_state = jnp.zeros((bs, d_state))
    action = jnp.zeros((bs,), dtype=jnp.int32)
    rng, *_rng = jax.random.split(rng, 1 + bs)
    params_mdp = jax.vmap(mdp.init)(jnp.stack(_rng), env_state, action)
    print(jax.tree_map(lambda x: x.shape, params_mdp))

    # ------------------------------------------------------------------------
    env_state = jnp.zeros((bs, d_state))

    obs = jnp.zeros((bs, d_obs))
    action = jnp.zeros((bs,), dtype=jnp.int32)
    reward = jnp.zeros((bs,))
    time = jnp.zeros((bs,), dtype=jnp.int32)
    agent_state = net.get_init_state(bs)

    print(env_state.shape, obs.shape, action.shape, reward.shape, time.shape)
    print(jax.tree_map(lambda x: x.shape, agent_state))

    forward_parallel = jax.vmap(partial(net.apply, params, method=net.forward_parallel))
    forward_recurrent = jax.vmap(partial(net.apply, params, method=net.forward_recurrent))
    transition_fn = partial(jax.vmap(partial(mdp.apply, method=mdp.get_transition)), params_mdp)
    obs_rew_done_fn = partial(jax.vmap(partial(mdp.apply, method=mdp.get_obs_rew_done)), params_mdp)

    # for i in range(10):
    #     agent_state, (logits, values) = forward_recurrent(agent_state, (obs, action, reward, time))
    #     rng, _rng = jax.random.split(rng)
    #     action = jax.random.categorical(_rng, logits=logits)
    #     env_state, (obs, reward) = jax.vmap(mdp.apply)(params_mdp, env_state, action)

    # def collect_step(carry, _):
    #     rng, agent_state_t, env_state_t, (act_tp, rew_tp, time_tp) = carry
    #     time_t = time_tp + 1
    #     obs_t, rew_t = get_obs_rew(env_state_t)
    #
    #     agent_state_tn, (logits_t, val_t) = forward_recurrent(agent_state_t, (obs_t, act_tp, rew_tp, time_t))
    #     rng, _rng = jax.random.split(rng)
    #     act_t = jax.random.categorical(_rng, logits=logits_t)
    #     env_state_tn = get_transition(env_state_t, act_t)
    #
    #     carry = rng, agent_state_tn, env_state_tn, (act_t, rew_t, time_t)
    #     return carry, dict(env_state=env_state_t, obs=obs_t, act_tp=act_tp, act=act_t, rew_tp=rew_tp, rew=rew_t,
    #                        time=time_t, logits=logits_t, val=val_t, done=jnp.zeros_like(time_t))

    # carry = rng, agent_state, env_state, (action, reward, time)
    # carry, buffer = jax.lax.scan(collect_step, carry, jnp.arange(30))
    # rng, agent_state, env_state, (action, reward, time) = carry

    agent_state, env_state, buffer, val_last = collect(rng, agent_state, env_state,
                                                       forward_recurrent, transition_fn, obs_rew_done_fn,
                                                       action, reward, time)

    print(jax.tree_map(lambda x: x.shape, agent_state))
    print(jax.tree_map(lambda x: x.shape, buffer))
    print(val_last.shape)

    # buffer = jax.tree_map(lambda x: rearrange(x, 't b ... -> b t ...'), buffer)
    # logits, vals = forward_parallel(buffer['obs'], buffer['act_tp'], buffer['rew_tp'], buffer['time'])
    # assert jnp.allclose(vals, buffer['val'], atol=1e-3)
    # assert jnp.allclose(logits, buffer['logits'], atol=1e-3)

    adv, ret = calc_gae(buffer, val_last)
    buffer.update(adv=adv, ret=ret)
    print(jax.tree_map(lambda x: x.shape, buffer))

    buffer = jax.tree_map(lambda x: rearrange(x, 't b ... -> b t ...'), buffer)

    tx = optax.chain(optax.clip_by_global_norm(1.), optax.adam(1e-2, eps=1e-5), )
    train_state = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

    # def update_step(train_state, _):
        # rng, _rng = jax.random.split(rng)
        # idx = jax.random.permutation(_rng, bs)
        # idx = jnp.arange(bs)
        # batch = jax.tree_map(lambda x: x[idx], buffer)
        # def _loss_fn(params, batch):
    #         forward_parallel = jax.vmap(partial(net.apply, params, method=net.forward_parallel))
    #         logits, value = forward_parallel(batch['obs'], batch['act_tp'], batch['rew_tp'], batch['time'])
    #
    #         log_prob = pi.log_prob(traj_batch.action)
    #
    #         value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
    #         value_losses = jnp.square(value - targets)
    #         value_losses_clipped = jnp.square(value_pred_clipped - targets)
    #         value_loss = (0.5 * jnp.maximum(value_losses, value_losses_clipped).mean())
    #
    #         ratio = jnp.exp(log_prob - traj_batch.log_prob)
    #         gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    #         loss_actor1 = ratio * gae
    #         loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"], ) * gae
    #         loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    #         loss_actor = loss_actor.mean()
    #         entropy = pi.entropy().mean()
    #
    #         total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
    #         return total_loss, (value_loss, loss_actor, entropy)
    #
    #     grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
    #     total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
    #     train_state = train_state.apply_gradients(grads=grads)
    #     return train_state, total_loss

    # carry = (train_state, traj_batch, advantages, targets, rng)
    # update_state, loss_info = jax.lax.scan(update_step, update_state, None, config["UPDATE_EPOCHS"])


if __name__ == "__main__":
    config = {
        'lr': 1e-3,
        'bs': 32,
        'n_steps': 128,
        'n_actions': 10,
        'd_state': 8,
        'd_obs': 32,
        'd_embd': 128,
    }
    rng = jax.random.PRNGKey(30)
    train(rng, config)


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
