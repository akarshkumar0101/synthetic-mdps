import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
import distrax
from einops import rearrange
from functools import partial

import mdps.wrappers


def make_train(config, env, network, callback=None, reset_env_iter=False, return_metric='rew'):
    config["NUM_UPDATES"] = (config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
    config["MINIBATCH_SIZE"] = (config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"])

    def linear_schedule(count):
        frac = (1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"])
        return config["LR"] * frac

    def train(md, rng, network_params):
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        env_params = jax.vmap(env.sample_params)(jax.random.split(_rng, config["NUM_ENVS"]))
        rng, _rng = jax.random.split(rng)
        obs, env_state = jax.vmap(env.reset, in_axes=(0, 0))(jax.random.split(_rng, config["NUM_ENVS"]), env_params)
        act_p, rew_p = jnp.zeros(config["NUM_ENVS"], dtype=jnp.int32), jnp.zeros(config["NUM_ENVS"])
        oar = obs, act_p, rew_p

        # INIT NETWORK
        forward_recurrent = jax.vmap(partial(network.apply, method=network.forward_recurrent),
                                     in_axes=(None, 0, (0, 0, 0)))
        forward_parallel = jax.vmap(partial(network.apply, method=network.forward_parallel),
                                    in_axes=(None, 0, 0, 0))

        rng, _rng = jax.random.split(rng)
        _rng = jax.random.split(_rng, config["NUM_ENVS"])
        agent_init_state = jax.vmap(network.get_init_state)(_rng)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx, )

        # TRAIN LOOP
        def _update_step(runner_state, _):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, _):
                i_iter, train_state, env_params, env_state, agent_state, (obs, act_p, rew_p), rng = runner_state

                # SELECT ACTION
                agent_state_n, (logits, value) = forward_recurrent(train_state.params,
                                                                   agent_state, (obs, act_p, rew_p))
                rng, _rng = jax.random.split(rng)
                pi = distrax.Categorical(logits=logits)
                act = pi.sample(seed=_rng)
                log_prob = pi.log_prob(act)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                _rng = jax.random.split(_rng, config["NUM_ENVS"])
                obs_n, env_state_n, rew, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, 0))(_rng, env_state, act,
                                                                                               env_params)
                runner_state = (i_iter, train_state, env_params, env_state_n, agent_state_n, (obs_n, act, rew), rng)
                transition = dict(obs=obs, act=act, rew=rew, done=done, info=info,
                                  act_p=act_p, rew_p=rew_p,
                                  logits=logits, log_prob=log_prob, val=value)
                return runner_state, transition

            i_iter, train_state, env_params, env_state, agent_state, (obs, act_p, rew_p), rng = runner_state
            agent_state = agent_init_state

            if reset_env_iter:
                # sample new env params
                rng, _rng = jax.random.split(rng)
                _rng = jax.random.split(_rng, config["NUM_ENVS"])
                env_params = jax.vmap(env.sample_params)(_rng)

                # reset environment but keep return statistics
                env_state_old = env_state
                rng, _rng = jax.random.split(rng)
                _rng = jax.random.split(_rng, config["NUM_ENVS"])
                obs, env_state = jax.vmap(env.reset, in_axes=(0, 0))(_rng, env_params)
                # if isinstance(env_state, mdps.wrappers.LogEnvState):
                #     env_state = mdps.wrappers.LogEnvState(env_state.env_state, 0., 0.,
                #                                           env_state_old.returned_episode_returns,
                #                                           env_state_old.returned_episode_lengths, env_state_old.timestep)

            runner_state = i_iter, train_state, env_params, env_state, agent_state, (obs, act_p, rew_p), rng
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])

            # CALCULATE ADVANTAGE
            i_iter, train_state, env_params, env_state, agent_state, (obs, act_p, rew_p), rng = runner_state
            # _, (_, last_val) = forward_recurrent(train_state.params, agent_state, (obs, act_p, rew_p))
            last_val = traj_batch['val'][-1]

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition["done"], transition["val"], transition["rew"]
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae)
                    return (gae, value), gae

                _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val),
                                             traj_batch, reverse=True, unroll=16, )
                return advantages, advantages + traj_batch['val']

            traj_batch['adv'], traj_batch['ret'] = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_batch(update_state, _):
                train_state, traj_batch, rng = update_state
                batch = traj_batch

                rng, _rng = jax.random.split(rng)
                n_envs_batch = config["NUM_ENVS"] // config["NUM_MINIBATCHES"]
                idx_env = jax.random.permutation(_rng, config["NUM_ENVS"])[:n_envs_batch]
                batch = jax.tree_util.tree_map(lambda x: x[:, idx_env], batch)
                # batch = jax.tree_util.tree_map( lambda x: rearrange(x, 't n ... -> (t n) ...'), batch )
                batch = jax.tree_util.tree_map(lambda x: rearrange(x, 't n ... -> n t ...'), batch)

                def _loss_fn(params, batch):
                    # RERUN NETWORK
                    logits, value = forward_parallel(params, batch['obs'], batch['act_p'], batch['rew_p'])
                    pi = distrax.Categorical(logits=logits)
                    log_prob = pi.log_prob(batch['act'])

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = batch['val'] + (value - batch['val']).clip(-config["CLIP_EPS"],
                                                                                    config["CLIP_EPS"])
                    value_losses = jnp.square(value - batch['ret'])
                    value_losses_clipped = jnp.square(value_pred_clipped - batch['ret'])
                    value_loss = (0.5 * jnp.maximum(value_losses, value_losses_clipped).mean())

                    # CALCULATE ACTOR LOSS
                    ratio = jnp.exp(log_prob - batch['log_prob'])
                    gae = batch['adv']
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"], ) * gae)
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()
                    entropy = pi.entropy().mean()

                    total_loss = (loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy)
                    return total_loss, (value_loss, loss_actor, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(train_state.params, batch)
                train_state = train_state.apply_gradients(grads=grads)

                update_state = train_state, traj_batch, rng
                return update_state, total_loss

            update_state = (train_state, traj_batch, rng)
            update_state, loss_info = jax.lax.scan(_update_batch, update_state, None,
                                                   config["UPDATE_EPOCHS"] * config["NUM_MINIBATCHES"])
            train_state = update_state[0]
            metric = traj_batch['info']
            metric['rew'] = traj_batch['rew']
            metric = jax.tree_map(lambda x: x.mean(axis=-1), metric)
            rng = update_state[-1]

            if callback is not None:
                jax.debug.callback(callback, md, i_iter, traj_batch)

            runner_state = (i_iter + 1, train_state, env_params, env_state, agent_state, (obs, act_p, rew_p), rng)
            return runner_state, (traj_batch['rew'] if return_metric == 'rew' else None)

        agent_state = jax.tree_map(lambda x: x, agent_init_state)  # copy
        rng, _rng = jax.random.split(rng)
        runner_state = (0, train_state, env_params, env_state, agent_state, oar, _rng)
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metric}

    return train


def main():
    import gymnax
    from mdps.wrappers import FlattenObservationWrapper, LogWrapper
    from agents.basic import BasicAgent
    from agents.linear_transformer import LinearTransformerAgent
    from jax.random import split

    import matplotlib.pyplot as plt
    from tqdm.auto import tqdm

    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 4,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 1e5,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ENV_NAME": "CartPole-v1",
        "ANNEAL_LR": False,
        "DEBUG": True,
    }
    rng = jax.random.PRNGKey(0)
    n_seeds = 16

    env, env_params = gymnax.make('CartPole-v1')
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    env.sample_params = lambda rng: env_params

    network = BasicAgent(env.action_space(env_params).n)
    # network = LinearTransformerAgent(n_acts=env.action_space(env_params).n,
    #                                  n_steps=config['NUM_STEPS'], n_layers=1, n_heads=4, d_embd=128)

    mds = jnp.arange(n_seeds)
    rng, _rng = jax.random.split(rng)
    rngs = jax.random.split(_rng, n_seeds)

    rng, _rng = jax.random.split(rng)
    _rng = jax.random.split(_rng, n_seeds)
    init_obs, init_act, init_rew = jnp.zeros((128, 4)), jnp.zeros((128,), dtype=jnp.int32), jnp.zeros((128,))
    network_params = jax.vmap(partial(network.init, method=network.forward_parallel),
                              in_axes=(0, None, None, None))(_rng, init_obs, init_act, init_rew)
    print(jax.tree_map(lambda x: x.shape, network_params))

    rets = [[] for _ in range(n_seeds)]

    pbar = tqdm(total=config["TOTAL_TIMESTEPS"])

    def callback(md, i_iter, traj_batch):
        if md == 0:
            pbar.update(config["NUM_ENVS"] * config["NUM_STEPS"])
        rets[md].append(traj_batch['info']['returned_episode_returns'].mean())

    train_fn = make_train(config, env, network, callback=callback)
    train_fn = jax.jit(jax.vmap(train_fn))

    out = train_fn(mds, rngs, network_params)

    rets = jnp.array(rets)
    print(rets.shape)

    plt.plot(jnp.mean(rets, axis=0), c=[.5, 0, 0, 1], label='mean')
    plt.plot(jnp.median(rets, axis=0), c=[.5, 0, 0, 1], label='median')
    plt.plot(rets.T, c=[.5, 0, 0, .1])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
