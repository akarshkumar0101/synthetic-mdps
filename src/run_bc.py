import argparse
import json
import os
import pickle

import jax
import optax
from einops import rearrange, reduce
from einops import repeat
from jax import config as jax_config
from jax.random import split
from tqdm.auto import tqdm

from algos.bc_dr import BC
from run import create_env, create_agent
from util import tree_stack

jax_config.update("jax_debug_nans", True)

parser = argparse.ArgumentParser()
# experiment args
parser.add_argument("--n_seeds", type=int, default=1)

parser.add_argument("--env_id", type=str, default="name=CartPole-v1")
parser.add_argument("--agent_id", type=str, default="obs_embed=dense;name=linear_transformer;tl=500")

parser.add_argument("--run", type=str, default='train')
parser.add_argument("--load_dir", type=str, default=None)
parser.add_argument("--load_dir_teacher", type=str, default=None)
parser.add_argument("--save_dir", type=str, default=None)
parser.add_argument("--save_buffers", type=lambda x: x == 'True', default=False)
parser.add_argument("--save_agent_params", type=lambda x: x == 'True', default=False)

parser.add_argument("--ft_first_last_layers", type=lambda x: x == 'True', default=False)

parser.add_argument("--n_iters", type=int, default=10)
# ppo args
parser.add_argument("--n_envs", type=int, default=4)
parser.add_argument("--n_steps", type=int, default=128)
parser.add_argument("--n_updates", type=int, default=16)
parser.add_argument("--n_envs_batch", type=int, default=1)
parser.add_argument("--lr", type=float, default=2.5e-4)
parser.add_argument("--clip_grad_norm", type=float, default=0.5)


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if v == 'None':
            setattr(args, k, None)
    assert args.run in ['train', 'eval']
    args.n_updates = 0 if args.run == 'eval' else args.n_updates
    return args


def run(args):
    cfg = vars(args)
    print(f"Config: {args}")

    rng = jax.random.PRNGKey(0)

    env = create_env(args.env_id)
    agent_student = create_agent(args.agent_id, args.env_id, n_acts=env.action_space(None).n)
    agent_teacher = create_agent(args.agent_id, args.env_id, n_acts=env.action_space(None).n)
    with open(f'{args.load_dir_teacher}/agent_params.pkl', 'rb') as f:
        agent_params_teacher = pickle.load(f)
        agent_params_teacher = jax.tree_map(lambda x: x[0], agent_params_teacher)

    # ft_transform=finetune_subset(["obs_embed", "actor", "critic"]) if args.ft_first_last_layers else optax.identity()
    ft_transform = optax.identity()
    tx = optax.chain(optax.clip_by_global_norm(args.clip_grad_norm), ft_transform, optax.adam(args.lr, eps=1e-5))

    bc = BC(agent_student, agent_teacher, agent_params_teacher, env, sample_env_params=env.sample_params, tx=tx,
            n_envs=args.n_envs, n_steps=args.n_steps, n_updates=args.n_updates, n_envs_batch=args.n_envs_batch,
            lr=args.lr, clip_grad_norm=args.clip_grad_norm, ent_coef=0.0)

    init_agent_env = jax.vmap(bc.init_agent_env)
    bc_step = jax.jit(jax.vmap(bc.bc_step, in_axes=(0, None)))

    rng, _rng = split(rng)
    carry = init_agent_env(split(_rng, args.n_seeds))
    if args.load_dir is not None:
        agent_state_student, carry_student, carry_teacher = carry
        rng, train_state, env_params, agent_state, obs, env_state = carry_student
        agent_params_new = train_state.params
        with open(f'{args.load_dir}/agent_params.pkl', 'rb') as f:
            agent_params_load = pickle.load(f)
            agent_params_load = jax.tree_map(lambda x: repeat(x, '1 ... -> n ...', n=args.n_seeds), agent_params_load)

        agent_params = agent_params_load

        # if args.ft_first_last_layers:
        if True:
            agent_params['params']['obs_embed'] = agent_params_new['params']['obs_embed']
            agent_params['params']['actor'] = agent_params_new['params']['actor']
            agent_params['params']['critic'] = agent_params_new['params']['critic']

        train_state = train_state.replace(params=agent_params)
        carry_student = rng, train_state, env_params, agent_state, obs, env_state
        carry = agent_state_student, carry_student, carry_teacher

    rets_student, rets_teacher, buffers = [], [], []
    pbar = tqdm(range(args.n_iters))
    for i_iter in pbar:
        carry, (buffer_student, buffer_teacher) = bc_step(carry, None)

        ret_student = reduce(buffer_student['info']['returned_episode_returns'], 's t e -> s', reduction='mean')
        ret_teacher = reduce(buffer_teacher['info']['returned_episode_returns'], 's t e -> s', reduction='mean')
        pbar.set_postfix(ret_student=ret_student.mean(), ret_teacher=ret_teacher.mean())
        rets_student.append(ret_student)
        rets_teacher.append(ret_teacher)
        if args.save_buffers:
            buffers.append(buffer_student)

    agent_state_student, carry_student, carry_teacher = carry
    rng, train_state, env_params, agent_state, obs, env_state = carry_student

    rets_student = rearrange(rets_student, 'n s -> s n')  # n_seeds, n_iters
    rets_teacher = rearrange(rets_teacher, 'n s -> s n')  # n_seeds, n_iters

    if len(buffers) > 0:
        buffers = tree_stack(buffers)

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(f'{args.save_dir}/config.json', 'w') as f:
            json.dump(cfg, f, indent=4)
        with open(f'{args.save_dir}/rets_student.pkl', 'wb') as f:
            pickle.dump(rets_student, f)
        with open(f'{args.save_dir}/rets_teacher.pkl', 'wb') as f:
            pickle.dump(rets_teacher, f)

        if args.save_agent_params:
            with open(f'{args.save_dir}/agent_params.pkl', 'wb') as f:
                pickle.dump(train_state.params, f)
        if args.save_buffers:
            with open(f'{args.save_dir}/buffers.pkl', 'wb') as f:
                pickle.dump(buffers, f)
    return rets_student, rets_teacher


if __name__ == "__main__":
    run(parse_args())
