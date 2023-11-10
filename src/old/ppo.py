import flax.linen
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from flax.training.train_state import TrainState

from functools import partial
from agent import Transformer
# from mdps import LinearSyntheticMDP

from wrappers import FlattenObservationWrapper, LogWrapper

import gymnax


def make_train(config):
    env, env_params = gymnax.make("CartPole-v1")
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    def train(rng, config):
        agent = Transformer(n_actions=env.action_space(env_params).n, n_steps=config['n_steps'],
                            n_layers=3, n_heads=4, d_embd=32)
        obs_init = jnp.zeros((config['n_steps'], *env.observation_space(env_params).shape))
        act_init = jnp.zeros((config['n_steps']), dtype=jnp.int32)
        rew_init = jnp.zeros((config['n_steps']))
        time_init = jnp.zeros((config['n_steps']), dtype=jnp.int32)
        rng, _rng = jax.random.split(rng)
        agent_params = agent.init(_rng, obs_init, act_init, rew_init, time_init)
        agent_state = agent.get_init_state(config['n_envs'])
        print('agent_params: ')
        print(jax.tree_map(lambda x: x.shape, agent_params))
        print('agent_state: ')
        print(jax.tree_map(lambda x: x.shape, agent_state))
        tx = optax.chain(optax.clip_by_global_norm(1.), optax.adam(config['lr'], eps=1e-5), )
        train_state = TrainState.create(apply_fn=agent.apply, params=agent_params, tx=tx, )

        rng, *_rng = jax.random.split(rng, 1 + config['n_envs'])
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(jnp.stack(_rng), env_params)
        act = jnp.zeros((config['n_envs'],), dtype=jnp.int32)
        rew = jnp.zeros((config['n_envs'],))
        time = jnp.zeros((config['n_envs'],), dtype=jnp.int32)

        runner_state = rng, train_state, env_params, agent_state, env_state, (obs, act, rew, time)

        def ppo_step(runner_state, _):
            def rollout(rng, agent_params, env_params, agent_state, env_state, oart, n_steps):
                """ Batch with jax.vmap """

                def rollout_step(carry, _):
                    rng, agent_state, env_state, oart = carry
                    obs, act_p, rew_p, time = oart
                    rng, _rng_agent, _rng_env = jax.random.split(rng, 3)
                    agent_state, (logits, val) = agent.apply(agent_params, agent_state, (obs, act_p, rew_p, time),
                                                             method=agent.forward_recurrent)
                    act = jax.random.categorical(_rng_agent, logits=logits).astype(jnp.int32)
                    obs_n, env_state, rew, done, info = env.step(_rng_env, env_state, act, env_params)
                    oart = obs_n, act, rew, time + 1
                    carry = rng, agent_state, env_state, oart
                    return carry, dict(obs=obs, act=act, rew=rew, done=done,
                                       time=time, logits=logits, val=val, act_p=act_p, rew_p=rew_p, info=info)

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

            batch_rollout = jax.vmap(rollout, in_axes=(0, None, None, 0, 0, 0, None))
            batch_calc_gae = jax.vmap(calc_gae, in_axes=(0, 0))

            rng, train_state, env_params, agent_state, env_state, oart = runner_state
            agent_state = agent.get_init_state(config['n_envs'])
            obs, act, rew, time = oart
            time = jnp.zeros((config['n_envs'],), dtype=jnp.int32)
            oart = obs, act, rew, time

            rng, *_rng = jax.random.split(rng, 1 + config['n_envs'])
            carry, buffer = batch_rollout(jnp.stack(_rng), train_state.params, env_params,
                                          agent_state, env_state, oart, config['n_steps'])
            _, agent_state, env_state, oart = carry

            print('buffer: ')
            print(jax.tree_map(lambda x: x.shape, buffer))

            # jax.debug.callback(lambda x: print(x['time']), buffer)
            # jax.debug.print('time: {time}', buffer['time'])

            val_last = buffer['val'][..., 0]
            adv, ret = batch_calc_gae(buffer, val_last)
            buffer.update(adv=adv, ret=ret)

            # jax.vmap(agent.apply)(train_state.params, buffer['obs'], buffer['act_p'], buffer['rew_p'], buffer['time')

            def update_step(runner_state, _):
                def loss_fn(params, batch):
                    print('loss_fn single example: ')
                    print(jax.tree_map(lambda x: x.shape, batch))
                    forward = partial(agent.apply, method=agent.forward_parallel)
                    logits, val = forward(params, batch['obs'], batch['act_p'], batch['rew_p'], batch['time'])
                    logits_old, val_old = batch['logits'], batch['val']
                    act = batch['act']
                    adv, ret = batch['adv'], batch['ret']
                    log_prob_old = jax.nn.log_softmax(logits_old)[jnp.arange(config['n_steps']), act]
                    log_prob = jax.nn.log_softmax(logits)[jnp.arange(config['n_steps']), act]

                    value_pred_clipped = val_old + (val - val_old).clip(-config['clip_coef'], config['clip_coef'])
                    value_losses = jnp.square(val - ret)
                    value_losses_clipped = jnp.square(value_pred_clipped - ret)
                    value_loss = (0.5 * jnp.maximum(value_losses, value_losses_clipped).mean())

                    ratio = jnp.exp(log_prob - log_prob_old)
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    loss_actor1 = ratio * adv
                    loss_actor2 = jnp.clip(ratio, 1.0 - config['clip_coef'], 1.0 + config['clip_coef'], ) * adv
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()

                    pi = jax.nn.softmax(logits)
                    logpi = jax.nn.log_softmax(logits)
                    entropy = -jnp.sum(pi * logpi, axis=-1).mean()

                    loss = loss_actor + config['vf_coef'] * value_loss - config['ent_coef'] * entropy
                    return loss
                loss_fn_v = jax.vmap(loss_fn, in_axes=(None, 0))
                loss_fn = lambda params, batch: loss_fn_v(params, batch).mean()
                valgrad_fn = jax.value_and_grad(loss_fn)

                rng, train_state, env_params, agent_state, env_state, oart = runner_state
                rng, _rng = jax.random.split(rng)

                idx_batch = jax.random.permutation(_rng, config['n_envs'])[:config['bs']]
                batch = jax.tree_map(lambda x: x[idx_batch], buffer)
                loss, grads = valgrad_fn(train_state.params, batch)
                train_state = train_state.apply_gradients(grads=grads)

                runner_state = rng, train_state, env_params, agent_state, env_state, oart
                return runner_state, None

            if True:
                def callback(infos):
                    for i in range(config['n_envs']):
                        info = jax.tree_map(lambda x: x[i], infos)
                        return_values = info["returned_episode_returns"][info["returned_episode"]]
                        timesteps = info["timestep"][info["returned_episode"]] * config["n_envs"]
                        for t in range(len(timesteps)):
                            print(f"global step={timesteps[t]}, episodic return={return_values[t]}")
                jax.debug.callback(callback, buffer['info'])

            runner_state = rng, train_state, env_params, agent_state, env_state, oart
            runner_state, _ = jax.lax.scan(update_step, runner_state, jnp.arange(config['n_updates']))
            return runner_state, buffer['info']

        runner_state, info = jax.lax.scan(ppo_step, runner_state, jnp.arange(config['n_iters']))
        return runner_state, info
    return train


if __name__ == '__main__':
    config = dict(lr=3e-4, bs=4, n_envs=4, n_steps=128,
                  # d_state=8, d_obs=32, d_embd=128,
                  n_updates=16, n_iters=1000,
                  clip_coef=0.2, vf_coef=0.5, ent_coef=0.01)
    #
    rng = jax.random.PRNGKey(30)
    train = partial(make_train(config), config=config)
    train = jax.jit(jax.vmap(train))
    runner_state, info = train(jnp.stack(jax.random.split(rng, 2)))
    # rng, train_state, env_params, agent_state, env_state, (obs, act, rew, time) = runner_state

    print('info: ')
    print(jax.tree_map(lambda x: x.shape, info))

    a = info['returned_episode_returns']
    print(a.shape)

    import matplotlib.pyplot as plt
    plt.plot(jnp.mean(a, axis=(0, 2, 3)))
    plt.show()


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
