import jax
import jax.numpy as jnp
import flax.linen as nn

from functools import partial

import gymnax
from mdps.wrappers import FlattenObservationWrapper, LogWrapper
from agents.basic import BasicAgent
from agents.linear_transformer import LinearTransformerAgent
from jax.random import split

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from ppo_general import make_train

from mdps.gridworld import GridEnv
from mdps.discrete_smdp import DiscreteInit, DiscreteTransition, DiscreteObs, DiscreteReward
import mdps.discrete_smdp
from mdps.syntheticmdp import SyntheticMDP
from mdps.wrappers_mine import TimeLimit


def main():
    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 16 * 4,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 64*5e5,
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
    from jax.random import split

    rng = jax.random.PRNGKey(0)
    n_seeds = 8

    # env, env_params = gymnax.make('CartPole-v1')
    # env = FlattenObservationWrapper(env)
    # env = LogWrapper(env)
    # env.sample_params = lambda rng: env_params

    env = GridEnv(8, start_state='random')
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    model_init = DiscreteInit(64)
    model_trans = DiscreteTransition(64, initializer=nn.initializers.normal(stddev=100))
    model_obs = DiscreteObs(64, 64, initializer=mdps.discrete_smdp.eye)
    # model_obs = DiscreteObs(64, 64)
    model_rew = DiscreteReward(64)
    env_syn = SyntheticMDP(None, None, 4, model_init, model_trans, model_obs, model_rew)
    env_syn = TimeLimit(env_syn, 4)
    env_syn = LogWrapper(env_syn)

    # network = BasicAgent(env.action_space(None).n)
    network = LinearTransformerAgent(n_acts=env.action_space(None).n,
                                     n_steps=config['NUM_STEPS'], n_layers=1, n_heads=4, d_embd=128)

    mds = jnp.arange(n_seeds)
    rng, _rng = jax.random.split(rng)
    rngs = jax.random.split(_rng, n_seeds)

    rng, _rng = jax.random.split(rng)
    _rng = jax.random.split(_rng, n_seeds)
    init_obs, init_act, init_rew = jnp.zeros((128, 64)), jnp.zeros((128,), dtype=jnp.int32), jnp.zeros((128,))
    network_params = jax.vmap(partial(network.init, method=network.forward_parallel),
                              in_axes=(0, None, None, None))(_rng, init_obs, init_act, init_rew)
    print(jax.tree_map(lambda x: x.shape, network_params))

    # rets = [[] for _ in range(n_seeds)]
    rews = [[] for _ in range(n_seeds)]
    pbar = tqdm(total=config["TOTAL_TIMESTEPS"])

    def callback(md, i_iter, traj_batch):
        if md == 0:
            pbar.update(config["NUM_ENVS"] * config["NUM_STEPS"])
        # rets[md].append(traj_batch['info']['returned_episode_returns'].mean())
        rews[md].append(traj_batch['rew'].mean(axis=-1))

    train_fn = make_train(config, env_syn, network, callback=callback, reset_env_iter=True)
    train_fn = jax.jit(jax.vmap(train_fn))

    out = train_fn(mds, rngs, network_params)
    pbar.close()

    # rets = jnp.array(rets)
    rews = jnp.array(rews)  # (n_seeds, n_iters, n_steps)
    rets = rews.sum(axis=-1)  # (n_seeds, n_iters)
    rews_start, rews_end = rews[:, :10, :].mean(axis=1), rews[:, -10:, :].mean(axis=1)  # (n_seeds, n_steps)
    print(rets.shape)
    print(rews.shape, rews_start.shape, rews_end.shape)
    network_params_trained = out['runner_state'][1].params

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(jnp.mean(rets, axis=0), c=[.5, 0, 0, 1], label='mean')
    # plt.plot(jnp.median(rets, axis=0), c=[.5, 0, 0, .5], label='median')
    plt.plot(rets.T, c=[.5, 0, 0, .1])
    plt.title('returns vs env steps')
    plt.legend()

    plt.subplot(122)
    plt.plot(jnp.mean(rews_start, axis=0), c=[.5, 0, 0, 1], label='mean start of training')
    plt.plot(rews_start.T, c=[.5, 0, 0, .1])
    plt.plot(jnp.mean(rews_end, axis=0), c=[0, .5, 0, 1], label='mean end of training')
    plt.plot(rews_end.T, c=[0, .5, 0, .1])
    plt.title('reward vs in-context steps')
    plt.legend()

    plt.show()

    config['LR'] = 0.
    config['TOTAL_TIMESTEPS'] = 5e5

    rews = [[] for _ in range(n_seeds)]
    pbar = tqdm(total=config["TOTAL_TIMESTEPS"])

    def callback(md, i_iter, traj_batch):
        if md == 0:
            pbar.update(config["NUM_ENVS"] * config["NUM_STEPS"])
        rews[md].append(traj_batch['rew'].mean(axis=-1))

    train_fn = make_train(config, env, network, callback=callback, reset_env_iter=True)
    train_fn = jax.jit(jax.vmap(train_fn))
    out = train_fn(mds, rngs, network_params_trained)

    rews = jnp.array(rews)  # (n_seeds, n_iters, n_steps)

    # plt.subplot(122)
    plt.plot(jnp.mean(rews.mean(axis=1), axis=0), c=[.5, 0, 0, 1], label='mean end of training')
    plt.plot(rews.mean(axis=1).T, c=[.5, 0, 0, .1])
    plt.title('reward vs in-context steps')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()


"""
Transfer tasks:
- 
"""