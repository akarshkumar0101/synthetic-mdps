import argparse
import os
import pickle
import json
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.random import split
from tqdm.auto import tqdm

from agents.basic import BasicAgent, RandomAgent
from agents.linear_transformer import LinearTransformerAgent
from mdps.continuous_smdp import ContinuousInit, ContinuousMatrixTransition, ContinuousMatrixObs, ContinuousReward
from mdps.discrete_smdp import DiscreteInit, DiscreteTransition, DiscreteObs, DiscreteReward
from mdps.gridworld import GridEnv
from mdps.natural_mdps import CartPole, MountainCar, Acrobot
from mdps.syntheticmdp import SyntheticMDP
from mdps.wrappers import FlattenObservationWrapper, LogWrapper
from mdps.wrappers_mine import TimeLimit, RandomlyProjectObservation

parser = argparse.ArgumentParser()
# experiment args
parser.add_argument("--n_seeds", type=int, default=1)
parser.add_argument("--agent", type=str, default="linear_transformer")
parser.add_argument("--env", type=str, default="gridworld")

# parser.add_argument("--save_fig", type=str, default=None)
parser.add_argument("--load_dir", type=str, default=None)
parser.add_argument("--save_dir", type=str, default=None)
parser.add_argument("--train", type=str, default='True')


def create_env(env_str):
    """
    Pretraining Tasks:
    "env=dsmdp;n_states=64;n_acts=4;d_obs=64;rpo=64;tl=4"
    "env=dsmdp;n_states=64;n_acts=4;d_obs=64;rpo=64;tl=128"

    "env=csmdp;d_state=8;n_acts=4;d_obs=64;delta=F;rpo=64;tl=128"
    "env=csmdp;d_state=8;n_acts=4;d_obs=64;delta=T;rpo=64;tl=128"

    Transfer tasks:
    "env=gridenv;grid_len=8;fobs=T;rpo=64;tl=128"
    "env=cartpole;fobs=T;rpo=64;tl=128"
    "env=mountaincar;fobs=T;rpo=64;tl=128"
    "env=acrobot;fobs=T;rpo=64;tl=128"

    """
    config = dict([sub.split('=') for sub in env_str.split(';')])

    if config['env'] == "cartpole":
        env = CartPole()
    elif config['env'] == "mountaincar":
        env = MountainCar()
    elif config['env'] == "acrobot":
        env = Acrobot()
    elif config['env'] == 'gridenv':
        grid_len = int(config['grid_len'])
        env = GridEnv(grid_len, start_state='random')

    elif config['env'] == 'dsmdp':
        n_states, d_obs, n_acts = int(config['n_states']), int(config['d_obs']), int(config['n_acts'])
        if config['rdist'] == 'U':
            def rdist(key, shape, dtype=float):
                return jax.random.uniform(key, shape, dtype=dtype, minval=-jnp.sqrt(3), maxval=jnp.sqrt(3))
        elif config['rdist'] == 'N':
            rdist = nn.initializers.normal(stddev=1.)
        # elif config['rdist'] == 'U.24':
        #     rdist = nn.initializers.uniform(scale=.24)

        model_init = DiscreteInit(n_states)
        model_trans = DiscreteTransition(n_states, initializer=nn.initializers.normal(stddev=100))
        model_obs = DiscreteObs(n_states, d_obs)
        model_rew = DiscreteReward(n_states, initializer=rdist)
        env = SyntheticMDP(n_acts, model_init, model_trans, model_obs, model_rew)
    elif config['env'] == 'csmdp':
        d_state, d_obs, n_acts = int(config['d_state']), int(config['d_obs']), int(config['n_acts'])
        delta = config['delta'] == 'T'
        model_init = ContinuousInit(d_state)
        model_trans = ContinuousMatrixTransition(d_state, delta)
        model_obs = ContinuousMatrixObs(d_state, d_obs)
        model_rew = ContinuousReward(d_state)
        env = SyntheticMDP(n_acts, model_init, model_trans, model_obs, model_rew)
    else:
        raise NotImplementedError
    if 'tl' in config:
        env = TimeLimit(env, int(config['tl']))  # this has to go before other ones because environment, not wrapper

    if 'fobs' in config and config['fobs'] == 'T':
        env = FlattenObservationWrapper(env)
    if 'rpo' in config and int(config['rpo']) > 0:
        env = RandomlyProjectObservation(env, d_obs=int(config['rpo']))

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


from algos.ppo.ppo_fixed_episode import make_ppo_funcs


def train(args):
    pass


def eval(args):
    config = vars(args)
    rng = jax.random.PRNGKey(0)

    env = create_env(args.env)
    agent = create_agent(env, args.agent, args.n_steps)

    init_agent_env, ppo_step, eval_step = make_ppo_funcs(agent, env, 4, 128, 16, 1, 2.5e-4)
    init_agent_env = jax.vmap(init_agent_env)
    ppo_step = jax.jit(jax.vmap(ppo_step, in_axes=(0, None)))
    eval_step = jax.jit(jax.vmap(eval_step, in_axes=(0, None)))

    rng, _rng = split(rng)
    carry = init_agent_env(split(_rng, args.n_seeds))

    if args.load_dir is not None:
        rng, train_state, env_params, agent_state, obs, env_state = carry
        with open(f'{args.load_dir}/train_state.pkl', 'rb') as f:
            train_state = pickle.load(f)
        carry = rng, train_state, env_params, agent_state, obs, env_state

    env_state, rew = [], []
    for _ in tqdm(range(10)):
        carry, buffer = eval_step(carry, None)
        env_state.append(buffer['env_state'])
        rew.append(buffer['rew'])
    rew = jnp.stack(rew, axis=1)  # n_seeds, n_iters, n_steps, n_envs

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        # save config
        with open(f'{args.save_dir}/config.json', 'w') as f:
            json.dump(config, f)
        # save rews
        with open(f'{args.save_dir}/rew_init.pkl', 'wb') as f:
            pickle.dump(rew_init, f)
        with open(f'{args.save_dir}/rew.pkl', 'wb') as f:
            pickle.dump(rew, f)



