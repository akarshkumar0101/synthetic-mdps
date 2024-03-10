import argparse
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from jax import config as jax_config
from jax.random import split
from tqdm.auto import tqdm

import create_env
from agents.regular_transformer import BCTransformer
from util import save_pkl
from icl_bc_ed import sample_batch_from_dataset

jax_config.update("jax_debug_nans", True)

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_path', type=str, default=None)
parser.add_argument('--env_id', type=str, default=None)
parser.add_argument('--ckpt_path', type=str, default=None)
parser.add_argument('--n_iters', type=int, default=3100)

parser.add_argument('--save_dir', type=str, default=None)

parser.add_argument('--n_envs', type=int, default=8)

# Model arguments
group = parser.add_argument_group("model")
group.add_argument("--n_layers", type=int, default=4)
group.add_argument("--n_heads", type=int, default=8)
group.add_argument("--d_embd", type=int, default=256)
group.add_argument("--ctx_len", type=int, default=1024)
group.add_argument("--mask_type", type=str, default="causal")
group.add_argument("--d_obs_uni", type=int, default=64)
group.add_argument("--n_acts_uni", type=int, default=10)



def main(args):
    rng = jax.random.PRNGKey(0)

    env = create_env.create_env(args.env_id)
    rng, _rng = split(rng)
    env_params = env.sample_params(_rng)

    d_obs, = env.observation_space(env_params).shape
    n_acts = env.action_space(env_params).n

    agent = BCTransformer(n_acts=args.n_acts_uni, n_layers=args.n_layers, n_heads=args.n_heads,
                          d_embd=args.d_embd, n_steps=args.ctx_len, mask_type=args.mask_type)
    T = args.ctx_len

    with open(args.dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    obs_mean, obs_std = dataset['obs'].mean(axis=(0, 1)), dataset['obs'].std(axis=(0, 1))

    expert_batch = sample_batch_from_dataset(jax.random.PRNGKey(0), dataset, args.n_envs, args.ctx_len//2)

    with open(args.ckpt_path, 'rb') as f:
        agent_params = pickle.load(f)['params']

    rng = jax.random.PRNGKey(0)
    obs_mat = jax.random.orthogonal(rng, n=max(d_obs, args.d_obs_uni), shape=())[:args.d_obs_uni, :d_obs]
    rng = jax.random.PRNGKey(1)
    obs_mat_aug = jax.random.normal(rng, (args.d_obs_uni, args.d_obs_uni)) * jnp.sqrt(1. / args.d_obs_uni)

    def t_obs(obs):
        obs = (obs - obs_mean) / (obs_std + 1e-5)
        obs = obs @ obs_mat.T
        obs = obs @ obs_mat_aug.T
        return obs

    def t_act(act):
        return act

    env_step = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))
    agent_forward = jax.jit(jax.vmap(agent.apply, in_axes=(None, 0, 0)))

    obs, state = jax.vmap(env.reset, in_axes=(0, None))(split(_rng, args.n_envs), env_params)
    # obs = ((obs - obs_mean) / (obs_std + 1e-5)) @ obs_mat.T

    # x_obs, x_act = [obs], [act0]

    act0 = jnp.zeros((args.n_envs,), dtype=int)
    print(expert_batch['obs'].shape, expert_batch['act'].shape, expert_batch['logits'].shape)

    x_obs = [expert_batch['obs'][:, i] for i in range(512)]
    x_act = [expert_batch['act'][:, i] for i in range(512)]
    x_obs.append(obs)
    x_act.append(act0)
    x_obs, x_act = x_obs[-T:], x_act[-T:]

    print(jnp.stack(x_obs, axis=1).shape, jnp.stack(x_act, axis=1).shape)

    rews, dones, rets = [], [], []

    pbar = tqdm(range(args.n_iters))
    for t in pbar:
        len_obs = len(x_obs)
        if len_obs < T:
            x_obsv = jnp.stack(x_obs + [obs] * (T - len_obs), axis=1)
            x_actv = jnp.stack(x_act + [act0] * (T - len_obs), axis=1)
            x_obsv = t_obs(x_obsv)
            x_actv = t_act(x_actv)

            logits = agent_forward(agent_params, x_obsv, x_actv)
            # logits = jnp.zeros_like(logits)
            rng, _rng = split(rng)
            act = jax.random.categorical(_rng, logits[:, len_obs - 1, :n_acts])
        else:
            x_obsv, x_actv = jnp.stack(x_obs, axis=1), jnp.stack(x_act, axis=1)
            x_obsv = t_obs(x_obsv)
            x_actv = t_act(x_actv)
            # print(x_actv[0])
            logits = agent_forward(agent_params, x_obsv, x_actv)
            # logits = jnp.zeros_like(logits)
            rng, _rng = split(rng)
            act = jax.random.categorical(_rng, logits[:, -1, :n_acts])
            # act = logits[:, time, :2].argmax(axis=-1)

        rng, _rng = split(rng)
        obs, state, rew, done, info = env_step(split(_rng, args.n_envs), state, act, env_params)
        rets.append(info['returned_episode_returns'])
        rews.append(rew)
        dones.append(done)

        x_obs.append(obs)
        x_act[-1] = act
        x_act.append(act0)
        x_obs, x_act = x_obs[-T:], x_act[-T:]

        pbar.set_postfix(rets=rets[-1].mean())
        print(info['returned_episode_returns'])

    rews = jnp.stack(rews, axis=0)
    dones = jnp.stack(dones, axis=0)
    rets = jnp.stack(rets, axis=0)
    rets = np.asarray(rets[-500:, :]).mean(axis=0)

    save_pkl(args.save_dir, "rets", rets)
    print(f"Score: {rets.mean()}")


if __name__ == '__main__':
    main(parser.parse_args())
