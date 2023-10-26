import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
import distrax
from mdps.wrappers import LogWrapper, FlattenObservationWrapper
from einops import rearrange
from functools import partial

from mdps.gridworld import GridEnv

from agents.linear_transformer import LinearTransformerAgent

global_iter = 0


def make_train(config, env, network):
    config["NUM_UPDATES"] = (config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
    config["MINIBATCH_SIZE"] = (config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"])

    def linear_schedule(count):
        frac = (1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"])
        return config["LR"] * frac

    def train(id, rng, network_params):
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
                train_state, env_params, env_state, agent_state, (obs, act_p, rew_p), rng = runner_state

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
                runner_state = (train_state, env_params, env_state_n, agent_state_n, (obs_n, act, rew), rng)
                transition = dict(obs=obs, act=act, rew=rew, done=done, info=info,
                                  act_p=act_p, rew_p=rew_p,
                                  logits=logits, log_prob=log_prob, val=value)
                return runner_state, transition

            train_state, env_params, env_state, agent_state, (obs, act_p, rew_p), rng = runner_state
            agent_state = agent_init_state

            rng, _rng = jax.random.split(rng)
            _rng = jax.random.split(_rng, config["NUM_ENVS"])
            env_params = jax.vmap(env.sample_params)(_rng)

            # rng, _rng = jax.random.split(rng)
            # _rng = jax.random.split(_rng, config["NUM_ENVS"])
            # obs, env_state = jax.vmap(env.reset, in_axes=(0, 0))(_rng, env_params)

            runner_state = train_state, env_params, env_state, agent_state, (obs, act_p, rew_p), rng
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])

            # CALCULATE ADVANTAGE
            train_state, env_params, env_state, agent_state, (obs, act_p, rew_p), rng = runner_state
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
            if config.get("DEBUG"):
                def callback(id, info):
                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    timesteps = info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    # for t in range(len(timesteps)):
                    #     print(f"global step={timesteps[t]}, episodic return={return_values[t]}")
                    if id == 0:
                        global global_iter
                        global_iter += 1
                        print(global_iter * config["NUM_STEPS"] * config["NUM_ENVS"])

                jax.debug.callback(callback, id, traj_batch['info'])

            runner_state = (train_state, env_params, env_state, agent_state, (obs, act_p, rew_p), rng)
            return runner_state, metric

        agent_state = jax.tree_map(lambda x: x, agent_init_state)  # copy
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_params, env_state, agent_state, oar, _rng)
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metric}

    return train


def main():
    from mdps.wrappers_mine import TimeLimit
    import gymnax
    from agents.basic import BasicAgent
    import matplotlib.pyplot as plt
    from mdps.discrete_smdp import DiscreteInit, DiscreteTransition, DiscreteObs, DiscreteReward
    from mdps.syntheticmdp import SyntheticMDP
    import flax.linen as nn

    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 16*4,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 100e6,
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
    n_seeds = 8

    # env, env_params = gymnax.make('CartPole-v1')
    # env = FlattenObservationWrapper(env)
    # env = LogWrapper(env)
    # env.sample_params = lambda rng: env_params

    # env = GridEnv(grid_len=8)
    # env = TimeLimit(env, 128)
    # env = FlattenObservationWrapper(env)
    # env = LogWrapper(env)
    # env_params = env.sample_params(rng)

    model_init = DiscreteInit(64)
    model_trans = DiscreteTransition(64, initializer=nn.initializers.normal(stddev=100))
    model_obs = DiscreteObs(64, 64)
    model_rew = DiscreteReward(64)
    env = SyntheticMDP(None, None, 4, model_init, model_trans, model_obs, model_rew)
    env = TimeLimit(env, 4)
    env = LogWrapper(env)
    env_params = env.sample_params(rng)

    # network = BasicAgent(env.action_space(env_params).n)
    network = LinearTransformerAgent(n_acts=env.action_space(env_params).n,
                                     n_steps=config['NUM_STEPS'], n_layers=1, n_heads=4, d_embd=128)

    train_fn = make_train(config, env, network)
    train_fn = jax.jit(jax.vmap(train_fn))

    ids = jnp.arange(n_seeds)
    rng, _rng = jax.random.split(rng)
    rngs = jax.random.split(_rng, n_seeds)

    rng, _rng = jax.random.split(rng)
    _rng = jax.random.split(_rng, n_seeds)

    init_obs, init_act, init_rew = jnp.zeros((128, 8*8)), jnp.zeros((128,), dtype=jnp.int32), jnp.zeros((128,))
    network_params = jax.vmap(partial(network.init, method=network.forward_parallel),
                              in_axes=(0, None, None, None))(_rng, init_obs, init_act, init_rew)
    network_params_init = network_params
    print(jax.tree_map(lambda x: x.shape, network_params))
    out = train_fn(ids, rngs, network_params)
    metrics = out['metrics']

    print(jax.tree_map(lambda x: x.shape, metrics))

    # metrics['returned_episode_returns']  # seed, n_iters, n_steps, n_envs
    # plt.plot(jnp.mean(metrics['returned_episode_returns'].mean(axis=(2, 3)), axis=0))
    # plt.plot(jnp.median(metrics['returned_episode_returns'].mean(axis=(2, 3)), axis=0))
    # plt.show()

    plt.plot(jnp.mean(metrics['rew'][:, :10].mean(axis=(1, )), axis=0))
    plt.plot(jnp.mean(metrics['rew'][:, -10:].mean(axis=(1, )), axis=0))
    plt.title('training')
    plt.show()

    # ----------------------------------------------------------------
    runner_state = out['runner_state']
    train_state = runner_state[0]
    network_params = train_state.params
    print(jax.tree_map(lambda x: x.shape, network_params))

    # now testing agent
    config['LR'] == 0
    config['TOTAL_TIMESTEPS'] = 1e6
    print(config)

    env = GridEnv(grid_len=8)
    env = TimeLimit(env, 128)
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    env_params = env.sample_params(rng)

    train_fn = make_train(config, env, network)
    train_fn = jax.jit(jax.vmap(train_fn))

    ids = jnp.arange(n_seeds)
    rng, _rng = jax.random.split(rng)
    rngs = jax.random.split(_rng, n_seeds)

    out = train_fn(ids, rngs, network_params_init)
    metrics = out['metrics']
    plt.plot(jnp.mean(metrics['rew'][:, :].mean(axis=(1, )), axis=0), label='init')

    out = train_fn(ids, rngs, network_params)
    metrics = out['metrics']
    plt.plot(jnp.mean(metrics['rew'][:, :].mean(axis=(1, )), axis=0), label='after pretraining')

    plt.title('testing')
    plt.legend()
    plt.show()

def main():
    from mdps.wrappers_mine import TimeLimit
    import gymnax
    from agents.basic import BasicAgent
    import matplotlib.pyplot as plt
    from mdps.discrete_smdp import DiscreteInit, DiscreteTransition, DiscreteObs, DiscreteReward
    from mdps.syntheticmdp import SyntheticMDP
    import flax.linen as nn

    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 16*4,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 100e6,
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
    n_seeds = 8

    # env, env_params = gymnax.make('CartPole-v1')
    # env = FlattenObservationWrapper(env)
    # env = LogWrapper(env)
    # env.sample_params = lambda rng: env_params

    # env = GridEnv(grid_len=8)
    # env = TimeLimit(env, 128)
    # env = FlattenObservationWrapper(env)
    # env = LogWrapper(env)
    # env_params = env.sample_params(rng)

    model_init = DiscreteInit(64)
    model_trans = DiscreteTransition(64, initializer=nn.initializers.normal(stddev=100))
    model_obs = DiscreteObs(64, 64)
    model_rew = DiscreteReward(64)
    env = SyntheticMDP(None, None, 4, model_init, model_trans, model_obs, model_rew)
    env = TimeLimit(env, 4)
    env = LogWrapper(env)
    env_params = env.sample_params(rng)

    # network = BasicAgent(env.action_space(env_params).n)
    network = LinearTransformerAgent(n_acts=env.action_space(env_params).n,
                                     n_steps=config['NUM_STEPS'], n_layers=1, n_heads=4, d_embd=128)

    train_fn = make_train(config, env, network)
    train_fn = jax.jit(jax.vmap(train_fn))

    ids = jnp.arange(n_seeds)
    rng, _rng = jax.random.split(rng)
    rngs = jax.random.split(_rng, n_seeds)

    rng, _rng = jax.random.split(rng)
    _rng = jax.random.split(_rng, n_seeds)

    init_obs, init_act, init_rew = jnp.zeros((128, 8*8)), jnp.zeros((128,), dtype=jnp.int32), jnp.zeros((128,))
    network_params = jax.vmap(partial(network.init, method=network.forward_parallel),
                              in_axes=(0, None, None, None))(_rng, init_obs, init_act, init_rew)
    network_params_init = network_params
    print(jax.tree_map(lambda x: x.shape, network_params))
    # out = train_fn(ids, rngs, network_params)
    # metrics = out['metrics']

    # print(jax.tree_map(lambda x: x.shape, metrics))

    # metrics['returned_episode_returns']  # seed, n_iters, n_steps, n_envs
    # plt.plot(jnp.mean(metrics['returned_episode_returns'].mean(axis=(2, 3)), axis=0))
    # plt.plot(jnp.median(metrics['returned_episode_returns'].mean(axis=(2, 3)), axis=0))
    # plt.show()

    # plt.plot(jnp.mean(metrics['rew'][:, :10].mean(axis=(1, )), axis=0))
    # plt.plot(jnp.mean(metrics['rew'][:, -10:].mean(axis=(1, )), axis=0))
    # plt.title('training')
    # plt.show()

    # ----------------------------------------------------------------
    # runner_state = out['runner_state']
    # train_state = runner_state[0]
    # network_params = train_state.params
    # print(jax.tree_map(lambda x: x.shape, network_params))

    # now testing agent
    config['LR'] == 0
    config['TOTAL_TIMESTEPS'] = 1e6
    print(config)

    env = GridEnv(grid_len=8)
    env = TimeLimit(env, 128)
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    env_params = env.sample_params(rng)

    train_fn = make_train(config, env, network)
    train_fn = jax.jit(jax.vmap(train_fn))

    ids = jnp.arange(n_seeds)
    rng, _rng = jax.random.split(rng)
    rngs = jax.random.split(_rng, n_seeds)

    out = train_fn(ids, rngs, network_params_init)
    metrics = out['metrics']
    plt.plot(jnp.mean(metrics['rew'][:, :].mean(axis=(1, )), axis=0), label='init')

    # out = train_fn(ids, rngs, network_params)
    # metrics = out['metrics']
    # plt.plot(jnp.mean(metrics['rew'][:, :].mean(axis=(1, )), axis=0), label='after pretraining')

    plt.title('testing')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
