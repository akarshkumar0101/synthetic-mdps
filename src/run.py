import argparse
import json
import os
import pickle
from functools import partial

import gymnax
import jax
import jax.numpy as jnp
import optax
from einops import rearrange, reduce
from jax import config as jax_config
from jax.random import split
from optax import GradientTransformation, EmptyState
from tqdm.auto import tqdm

from agents import BasicAgent, LinearTransformerAgent, DenseObsEmbed, ObsActRewTimeEmbed
from algos.ppo_dr import PPO
from mdps import smdp, csmdp
from mdps.wrappers import LogWrapper
from mdps.wrappers_mine import DoneObsActRew, MetaRLWrapper, TimeLimit, ObsNormRand
from util import tree_stack

jax_config.update("jax_debug_nans", True)

parser = argparse.ArgumentParser()
# experiment args
parser.add_argument("--n_seeds", type=int, default=1)

parser.add_argument("--env_id", type=str, default="name=CartPole-v1")
parser.add_argument("--agent_id", type=str, default="obs_embed=dense;name=linear_transformer;tl=500")

parser.add_argument("--run", type=str, default='train')
parser.add_argument("--load_dir", type=str, default=None)
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
parser.add_argument("--clip_eps", type=float, default=0.2)
parser.add_argument("--vf_coef", type=float, default=0.5)
parser.add_argument("--ent_coef", type=float, default=0.01)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae_lambda", type=float, default=0.95)


def finetune_subset(ft_keys) -> GradientTransformation:
    def init_fn(params):
        del params
        return EmptyState()

    def update_fn(updates, state, params=None):
        del params  # Unused by the zero transform.
        updates = jax.tree_map(lambda x: x, updates)  # copy
        for ft_key in ft_keys:
            assert ft_key in updates['params']
        for k in updates['params'].keys():
            if k not in ft_keys:
                updates['params'][k] = jax.tree_map(lambda x: jnp.zeros_like(x), updates['params'][k])
        return updates, state

    return GradientTransformation(init_fn, update_fn)


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

    if env_cfg['name'] in gymnax.registered_envs:
        env, env_params = gymnax.make(env_cfg['name'])
        env.sample_params = lambda rng: env_params
    elif env_cfg['name'] == 'csmdp':
        d_state, d_obs, n_acts = int(env_cfg['d_state']), int(env_cfg['d_obs']), int(env_cfg['n_acts'])
        delta = env_cfg['delta'] == 'T'
        model_init = csmdp.Init(d_state=d_state)
        if env_cfg['trans'] == 'linear':
            model_trans = csmdp.LinearTransition(d_state=d_state, n_acts=n_acts, delta=delta)
        elif env_cfg['trans'] == 'mlp':
            model_trans = csmdp.MLPTransition(d_state=d_state, n_acts=n_acts, delta=delta)
        else:
            raise NotImplementedError
        model_obs = csmdp.LinearObservation(d_state=d_state, d_obs=d_obs)
        if env_cfg['rew'] == 'linear':
            model_rew = csmdp.LinearReward(d_state=d_state)
        elif env_cfg['rew'] == 'goal':
            model_rew = csmdp.GoalReward(d_state=d_state)
        else:
            raise NotImplementedError
        model_done = smdp.NeverDone()
        env = smdp.SyntheticMDP(model_init, model_trans, model_obs, model_rew, model_done)
    else:
        raise NotImplementedError
    env = ObsNormRand(env)

    if "mrl" in env_cfg:
        n_trials, n_steps_trial = [int(x) for x in env_cfg['mrl'].split('x')]
        env = MetaRLWrapper(env, n_trials=n_trials, n_steps_trial=n_steps_trial)
    elif 'tl' in env_cfg:
        env = TimeLimit(env, n_steps_max=int(env_cfg['tl']))

    # if 'fobs' in env_cfg and env_cfg['fobs'] == 'T':
    #     env = FlattenObservationWrapper(env)
    # if 'rpo' in env_cfg and env_cfg['rpo'].isdigit():
    #     env = RandomlyProjectObservation(env, d_out=int(env_cfg['rpo']))
    # env = GaussianObsReward(env, n_envs=2048, n_steps=n_steps)

    env = DoneObsActRew(env)

    env = LogWrapper(env)
    return env


def create_agent(agent_id, env_id, n_acts):
    agent_cfg = dict([sub.split('=') for sub in agent_id.split(';')])
    env_cfg = dict([sub.split('=') for sub in env_id.split(';')])

    tl = int(agent_cfg['tl'])
    if 'MinAtar' in env_cfg['name']:
        # ObsEmbed = partial(MinAtarObsEmbed, d_embd=128)
        ObsEmbed = partial(DenseObsEmbed, d_embd=128)
    else:
        ObsEmbed = partial(DenseObsEmbed, d_embd=128)
    ObsEmbed = partial(ObsActRewTimeEmbed, d_embd=128, ObsEmbed=ObsEmbed, n_acts=n_acts, n_steps_max=tl)

    if agent_cfg['name'] == 'random':
        # agent = RandomAgent(n_acts)
        raise NotImplementedError
    elif agent_cfg['name'] == 'basic':
        agent = BasicAgent(ObsEmbed, n_acts)
    elif agent_cfg['name'] == 'linear_transformer':
        agent = LinearTransformerAgent(ObsEmbed, n_acts, n_layers=2, n_heads=4, d_embd=128)  # TODO: go back to this
        # agent = LinearTransformerAgent(ObsEmbed, n_acts, n_layers=4, n_heads=4, d_embd=128)
    else:
        raise NotImplementedError
    return agent


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
    agent = create_agent(args.agent_id, args.env_id, n_acts=env.action_space(None).n)

    ft_transform = finetune_subset(["obs_embed", "actor", "critic"]) if args.ft_first_last_layers else optax.identity()
    tx = optax.chain(optax.clip_by_global_norm(args.clip_grad_norm), ft_transform, optax.adam(args.lr, eps=1e-5))
    ppo = PPO(agent, env, sample_env_params=env.sample_params, tx=tx,
              n_envs=args.n_envs, n_steps=args.n_steps, n_updates=args.n_updates, n_envs_batch=args.n_envs_batch,
              lr=args.lr, clip_grad_norm=args.clip_grad_norm, clip_eps=args.clip_eps,
              vf_coef=args.vf_coef, ent_coef=args.ent_coef, gamma=args.gamma, gae_lambda=args.gae_lambda)
    init_agent_env = jax.vmap(ppo.init_agent_env)
    # eval_step = jax.jit(jax.vmap(ppo.eval_step, in_axes=(0, None)))
    ppo_step = jax.jit(jax.vmap(ppo.ppo_step, in_axes=(0, None)))

    rng, _rng = split(rng)
    carry = init_agent_env(split(_rng, args.n_seeds))
    if args.load_dir is not None:
        rng, train_state, env_params, agent_state, obs, env_state = carry
        agent_params_new = train_state.params
        with open(f'{args.load_dir}/agent_params.pkl', 'rb') as f:
            agent_params_load = pickle.load(f)

        agent_params = agent_params_load
        if args.ft_first_last_layers:
            agent_params['params']['obs_embed'] = agent_params_new['params']['obs_embed']
            agent_params['params']['actor'] = agent_params_new['params']['actor']
            agent_params['params']['critic'] = agent_params_new['params']['critic']

        train_state = train_state.replace(params=agent_params)
        carry = rng, train_state, env_params, agent_state, obs, env_state

    rets, buffers = [], []
    pbar = tqdm(range(args.n_iters))
    for i_iter in pbar:
        carry, buffer = ppo_step(carry, None)

        # rew = reduce(buffer['rew'], 's t e -> s t', reduction='mean')
        ret = reduce(buffer['info']['returned_episode_returns'], 's t e -> s', reduction='mean')
        pbar.set_postfix(ret=ret.mean())
        rets.append(ret)
        if args.save_buffers:
            buffers.append(buffer)
    rng, train_state, env_params, agent_state, obs, env_state = carry
    rets = rearrange(rets, 'n s -> s n')  # n_seeds, n_iters
    if len(buffers) > 0:
        buffers = tree_stack(buffers)

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(f'{args.save_dir}/config.json', 'w') as f:
            json.dump(cfg, f, indent=4)
        with open(f'{args.save_dir}/rets.pkl', 'wb') as f:
            pickle.dump(rets, f)

        if args.save_agent_params:
            with open(f'{args.save_dir}/agent_params.pkl', 'wb') as f:
                pickle.dump(train_state.params, f)
            best_idx = rets[:, -1].argmax()
            with open(f'{args.save_dir}/agent_params_best.pkl', 'wb') as f:
                pickle.dump(jax.tree_map(lambda x: x[best_idx], train_state.params), f)

        if args.save_buffers:
            with open(f'{args.save_dir}/buffers.pkl', 'wb') as f:
                pickle.dump(buffers, f)
    return rets


if __name__ == "__main__":
    run(parse_args())
