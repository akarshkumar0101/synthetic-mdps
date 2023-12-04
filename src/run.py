import argparse
import json
import os
import pickle

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import config
from jax.random import split
from tqdm.auto import tqdm

from agents.basic import BasicAgent, RandomAgent
from agents.linear_transformer import LinearTransformerAgent
from algos.ppo_fixed_episode import make_ppo_funcs
from mdps.continuous_smdp import ContinuousInit, ContinuousMatrixTransition, ContinuousMatrixObs, ContinuousReward
from mdps.discrete_smdp import DiscreteInit, DiscreteTransition, DiscreteObs, DiscreteReward
from mdps.gridenv import GridEnv
from mdps.natural_mdps import CartPole, MountainCar, Acrobot
from mdps.syntheticmdp import SyntheticMDP
from mdps.wrappers_mine import TimeLimit, RandomlyProjectObservation, DoneObsActRew, FlattenObservationWrapper, \
    GaussianReward, MetaRLWrapper

config.update("jax_debug_nans", True)

parser = argparse.ArgumentParser()
# experiment args
parser.add_argument("--n_seeds", type=int, default=1)
parser.add_argument("--env_id", type=str, default="name=cartpole;tl=128")
parser.add_argument("--agent_id", type=str, default="linear_transformer")

parser.add_argument("--run", type=str, default='train')
parser.add_argument("--load_dir", type=str, default=None)
parser.add_argument("--save_dir", type=str, default=None)

# parser.add_argument("--save_fig", type=str, default=None)
parser.add_argument("--n_iters", type=int, default=10)

parser.add_argument("--n_envs", type=int, default=4)  # 128
parser.add_argument("--n_steps", type=int, default=128)
parser.add_argument("--n_updates", type=int, default=16)
parser.add_argument("--n_envs_batch", type=int, default=1)  # 32
parser.add_argument("--lr", type=float, default=2.5e-4)
parser.add_argument("--clip_grad_norm", type=float, default=0.5)
parser.add_argument("--clip_eps", type=float, default=0.2)
parser.add_argument("--vf_coef", type=float, default=0.5)
parser.add_argument("--ent_coef", type=float, default=0.01)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae_lambda", type=float, default=0.95)


def create_env(env_id):
    """
    Pretraining Tasks:
    "name=dsmdp;n_states=64;n_acts=4;d_obs=64;rpo=64;tl=4"
    "name=dsmdp;n_states=64;n_acts=4;d_obs=64;rpo=64;tl=128"

    "name=csmdp;d_state=8;n_acts=4;d_obs=64;delta=F;rpo=64;tl=128"
    "name=csmdp;d_state=8;n_acts=4;d_obs=64;delta=T;rpo=64;tl=128"

    Transfer tasks:
    "name=gridenv;grid_len=8;fobs=T;rpo=64;tl=128"
    "name=cartpole;fobs=T;rpo=64;tl=128"
    "name=mountaincar;fobs=T;rpo=64;tl=128"
    "name=acrobot;fobs=T;rpo=64;tl=128"

    """
    env_cfg = dict([sub.split('=') for sub in env_id.split(';')])

    if env_cfg['name'] == "cartpole":
        env = CartPole()
        env = GaussianReward(env, n_envs=1024, n_steps=128)
    elif env_cfg['name'] == "mountaincar":
        env = MountainCar()
        env = GaussianReward(env, n_envs=1024, n_steps=128)
    elif env_cfg['name'] == "acrobot":
        env = Acrobot()
        env = GaussianReward(env, n_envs=1024, n_steps=128)
    elif env_cfg['name'] == 'gridenv':
        grid_len, pos_start, pos_rew = int(env_cfg['grid_len']), env_cfg['pos_start'], env_cfg['pos_rew']
        env = GridEnv(grid_len, pos_start=pos_start, pos_rew=pos_rew)
        env = GaussianReward(env, n_envs=1024, n_steps=128)
    elif env_cfg['name'] == 'dsmdp':
        n_states, d_obs, n_acts = int(env_cfg['n_states']), int(env_cfg['d_obs']), int(env_cfg['n_acts'])
        rdist = nn.initializers.normal(stddev=1.)
        # if env_cfg['rdist'] == 'U':
        #     def rdist(key, shape, dtype=float):
        #         return jax.random.uniform(key, shape, dtype=dtype, minval=-jnp.sqrt(3), maxval=jnp.sqrt(3))
        # elif env_cfg['rdist'] == 'N':
        #     rdist = nn.initializers.normal(stddev=1.)
        # elif config['rdist'] == 'U.24':
        #     rdist = nn.initializers.uniform(scale=.24)
        model_init = DiscreteInit(n_states)
        model_trans = DiscreteTransition(n_states, initializer=nn.initializers.normal(stddev=100))
        model_obs = DiscreteObs(n_states, d_obs)
        model_rew = DiscreteReward(n_states, initializer=rdist)
        env = SyntheticMDP(n_acts, model_init, model_trans, model_obs, model_rew)
    elif env_cfg['name'] == 'csmdp':
        d_state, d_obs, n_acts = int(env_cfg['d_state']), int(env_cfg['d_obs']), int(env_cfg['n_acts'])
        delta = env_cfg['delta'] == 'T'
        model_init = ContinuousInit(d_state)
        model_trans = ContinuousMatrixTransition(d_state, delta)
        model_obs = ContinuousMatrixObs(d_state, d_obs)
        model_rew = ContinuousReward(d_state)
        env = SyntheticMDP(n_acts, model_init, model_trans, model_obs, model_rew)
    else:
        raise NotImplementedError

    # if 'tl' in env_cfg and env_cfg['tl'].isdigit():
    #     env = TimeLimit(env, n_steps=int(env_cfg['tl']))
    env = MetaRLWrapper(env, n_steps_trial=16, n_trials=8)
    if 'fobs' in env_cfg and env_cfg['fobs'] == 'T':
        env = FlattenObservationWrapper(env)
    if 'rpo' in env_cfg and env_cfg['rpo'].isdigit():
        env = RandomlyProjectObservation(env, d_out=int(env_cfg['rpo']))
    env = DoneObsActRew(env)
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


def run(args):
    print(args)
    assert args.run in ['train', 'eval']
    config = vars(args)

    rng = jax.random.PRNGKey(0)

    env = create_env(args.env_id)
    agent = create_agent(env, args.agent_id, 128)

    init_agent_env, eval_step, ppo_step = make_ppo_funcs(
        agent, env, args.n_envs, args.n_steps, args.n_updates, args.n_envs_batch, args.lr, args.clip_grad_norm,
        args.clip_eps, args.vf_coef, args.ent_coef, args.gamma, args.gae_lambda
    )
    init_agent_env = jax.vmap(init_agent_env)
    eval_step = jax.jit(jax.vmap(eval_step, in_axes=(0, None)))
    ppo_step = jax.jit(jax.vmap(ppo_step, in_axes=(0, None)))

    rng, _rng = split(rng)
    carry = init_agent_env(split(_rng, args.n_seeds))
    if args.load_dir:
        rng, train_state, env_params, agent_state, obs, env_state = carry
        with open(f'{args.load_dir}/agent_params.pkl', 'rb') as f:
            agent_params = pickle.load(f)
        train_state = train_state.replace(params=agent_params)
        carry = rng, train_state, env_params, agent_state, obs, env_state
    step_fn = ppo_step if args.run == 'train' else eval_step
    buffers, rew = [], []
    pbar = tqdm(range(args.n_iters))
    for _ in pbar:
        carry, buffer = step_fn(carry, None)
        rew.append(buffer['rew'])
        if args.run == 'eval':
            buffers.append(buffer)
        pbar.set_postfix(rew=buffer['rew'].mean())
    rng, train_state, env_params, agent_state, obs, env_state = carry
    rew = jnp.stack(rew, axis=1)  # n_seeds, n_iters, n_steps, n_envs

    import matplotlib.pyplot as plt
    a = buffer['info']['done_trial'][0].T
    plt.imshow(a)
    plt.show()

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        # save config
        with open(f'{args.save_dir}/config.json', 'w') as f:
            json.dump(config, f, indent=4)

        with open(f'{args.save_dir}/rew.pkl', 'wb') as f:
            pickle.dump(rew, f)
        if args.run == 'train':
            with open(f'{args.save_dir}/agent_params.pkl', 'wb') as f:
                pickle.dump(train_state.params, f)
        elif args.run == 'eval':
            with open(f'{args.save_dir}/buffers.pkl', 'wb') as f:
                pickle.dump(buffers, f)


if __name__ == "__main__":
    run(parser.parse_args())
