import argparse
import os
import pickle
from functools import partial

import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.random import split
from tqdm.auto import tqdm

import mdps.discrete_smdp
from agents.basic import BasicAgent, RandomAgent
from agents.linear_transformer import LinearTransformerAgent
from algos.ppo_fixed_episode import make_train
from mdps.continuous_smdp import ContinuousInit, ContinuousMatrixTransition, ContinuousMatrixObs, ContinuousReward
from mdps.discrete_smdp import DiscreteInit, DiscreteTransition, DiscreteObs, DiscreteReward
from mdps.gridworld import GridEnv
from mdps.syntheticmdp import SyntheticMDP
from mdps.wrappers import FlattenObservationWrapper, LogWrapper
from mdps.wrappers_mine import TimeLimit, RandomlyProjectObservation
from mdps.mountain_car import MountainCar

parser = argparse.ArgumentParser()
# experiment args
parser.add_argument("--n_seeds", type=int, default=1)
parser.add_argument("--agent", type=str, default="linear_transformer")
parser.add_argument("--env", type=str, default="gridworld")

parser.add_argument("--save_fig", type=str, default=None)
parser.add_argument("--save_agent", type=str, default=None)

# PPO args
parser.add_argument("--lr", type=float, default=2.5e-4)
parser.add_argument("--n_envs", type=int, default=16 * 4)
parser.add_argument("--n_steps", type=int, default=128)
# parser.add_argument("--total_timesteps", type=float, default=1 * 16 * 5e5)
parser.add_argument("--total_timesteps", type=float, default=1 * 5e5)
parser.add_argument("--update_epochs", type=int, default=4)
parser.add_argument("--n_minibatches", type=int, default=4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--clip_eps", type=float, default=0.2)
parser.add_argument("--ent_coef", type=float, default=0.01)
parser.add_argument("--vf_coef", type=float, default=0.5)
parser.add_argument("--max_grad_norm", type=float, default=0.5)


def create_env(env_str):
    """
    "env=gridenv;grid_len=8;fobs=T;rpo=F;tl=128"

    "env=dsmdp;n_states=64;d_obs=64;n_acts=4;rpo=F;tl=128"

    """
    config = dict([sub.split('=') for sub in env_str.split(';')])

    if config['env'] == 'cartpole':
        env, env_params = gymnax.make('CartPole-v1')
        env.sample_params = lambda rng: env_params
    elif config['env'] == 'gridenv':
        grid_len = int(config['grid_len'])
        env = GridEnv(grid_len, start_state='random')
    elif config['env'] == 'mountaincar':
        env = MountainCar()
        env_params = env.default_params
        env.sample_params = lambda rng: env_params
    elif config['env'] == 'dsmdp':
        n_states, d_obs, n_acts = int(config['n_states']), int(config['d_obs']), int(config['n_acts'])
        model_init = DiscreteInit(n_states)
        model_trans = DiscreteTransition(n_states, initializer=nn.initializers.normal(stddev=100))
        model_obs = DiscreteObs(n_states, d_obs)
        model_rew = DiscreteReward(n_states)
        env = SyntheticMDP(n_acts, model_init, model_trans, model_obs, model_rew)
    elif config['env'] == 'csmdp':
        d_state, d_obs, n_acts = int(config['d_state']), int(config['d_obs']), int(config['n_acts'])
        model_init = ContinuousInit(d_state)
        model_trans = ContinuousMatrixTransition(d_state)
        model_obs = ContinuousMatrixObs(d_state, d_obs)
        model_rew = ContinuousReward(d_state)
        env = SyntheticMDP(n_acts, model_init, model_trans, model_obs, model_rew)
    else:
        raise NotImplementedError
    if 'tl' in config:
        env = TimeLimit(env, int(config['tl']))  # this has to go before other ones because it environemnt, not wrapper

    if 'fobs' in config and config['fobs'] == 'T':
        env = FlattenObservationWrapper(env)
    if 'rpo' in config and config['rpo'] == 'T':
        env = RandomlyProjectObservation(env)
    env = LogWrapper(env)
    return env


def create_agent(env, agent_name, n_steps):
    if agent_name == 'random':
        agent = RandomAgent(env.action_space(None).n)
    elif agent_name == 'basic':
        agent = BasicAgent(env.action_space(None).n)
    elif agent_name == 'linear_transformer':
        agent = LinearTransformerAgent(n_acts=env.action_space(None).n,
                                       n_steps=n_steps, n_layers=1, n_heads=4, d_embd=128)
    else:
        raise NotImplementedError
    return agent


def main(args):
    print(args)
    config = vars(args)
    config = {k.upper(): v for k, v in config.items()}
    config = {k.replace('N_', 'NUM_'): v for k, v in config.items()}
    config['ANNEAL_LR'] = False

    rng = jax.random.PRNGKey(0)
    n_seeds = args.n_seeds

    n_steps = args.n_steps
    d_obs = 64

    env = create_env(args.env)
    agent = create_agent(env, args.agent, n_steps)

    init_obs, init_act, init_rew = jnp.zeros((n_steps, d_obs)), jnp.zeros((n_steps,), dtype=jnp.int32), jnp.zeros(
        (n_steps,))
    rng, _rng = jax.random.split(rng)
    _rng = jax.random.split(_rng, n_seeds)
    agent_params = jax.vmap(partial(agent.init, method=agent.forward_parallel),
                            in_axes=(0, None, None, None))(_rng, init_obs, init_act, init_rew)

    # -------------------- PRETRAINING --------------------
    agent_params_trained, rews = train_agent(rng, config, env, agent, agent_params, n_seeds)
    # rews.shape is (n_seeds, n_iters, n_steps)
    rets = rews.sum(axis=-1)  # (n_seeds, n_iters)
    rews_start, rews_end = rews[:, :10, :].mean(axis=1), rews[:, -10:, :].mean(axis=1)  # (n_seeds, n_steps)

    if args.save_fig is not None:
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.plot(jnp.mean(rets, axis=0), c=[.5, 0, 0, 1], label='mean')
        # plt.plot(jnp.median(rets, axis=0), c=[.5, 0, 0, .5], label='median')
        plt.plot(rets.T, c=[.5, 0, 0, .1])
        plt.title('SyntheticEnv (pretraining)')
        plt.ylabel('Single Episode Return')
        plt.xlabel('Training Time')
        plt.legend()
        plt.subplot(122)
        plt.plot(jnp.mean(rews_start, axis=0), c=[.5, 0, 0, 1], label='mean start of training')
        plt.plot(rews_start.T, c=[.5, 0, 0, .1])
        plt.plot(jnp.mean(rews_end, axis=0), c=[0, .5, 0, 1], label='mean end of training')
        plt.plot(rews_end.T, c=[0, .5, 0, .1])
        plt.title('SyntheticEnv (pretraining)')
        plt.ylabel('Per-Timestep Reward')
        plt.xlabel('In-Context Timesteps')
        plt.legend()
        plt.show()

    if args.save_agent is not None:
        os.makedirs(os.path.dirname(args.save_agent), exist_ok=True)
        with open(args.save_agent, 'wb') as f:
            pickle.dump(agent_params_trained, f)


def init_agent_params(rng, agent, n_seeds, n_steps, d_obs):
    pass


def callback_pbar(md, i_iter, traj_batch, pbar=None, n_envs=0, n_steps=0):
    if md == 0 and pbar is not None:
        pbar.update(n_envs * n_steps)


def train_agent(rng, config, env, agent, agent_params_init, n_seeds):
    pbar = tqdm(total=config['TOTAL_TIMESTEPS'])
    callback = partial(callback_pbar, pbar=pbar, n_envs=config['NUM_ENVS'], n_steps=config['NUM_STEPS'])

    train_fn = make_train(config, env, agent, callback=callback, reset_env_iter=True, return_metric='rew_mean')
    # train_fn = jax.jit(jax.vmap(train_fn))
    # devs = jax.devices()
    # devs = [devs[i % len(devs)] for i in range(n_seeds)]
    train_fn = jax.pmap(train_fn)

    mds = jnp.arange(n_seeds)
    rng, _rng = jax.random.split(rng)
    out = train_fn(mds, split(_rng, n_seeds), agent_params_init)
    pbar.close()

    agent_params_final = out['runner_state'][1].params
    rews = out['metrics']  # (n_seeds, n_iters, n_steps)
    return agent_params_final, rews


def eval_agent(rng, config, env, agent, agent_params_init, n_seeds):
    config = config.copy()
    config['LR'] = 0.
    config['TOTAL_TIMESTEPS'] = 1e6
    config['UPDATE_EPOCHS'] = 0
    _, rews = train_agent(rng, config, env, agent, agent_params_init, n_seeds)
    return rews


if __name__ == '__main__':
    main(parser.parse_args())
