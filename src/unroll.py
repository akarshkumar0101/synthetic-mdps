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

jax_config.update("jax_debug_nans", True)

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_path', type=str, default=None)
parser.add_argument('--env_id', type=str, default=None)
parser.add_argument('--ckpt_path', type=str, default=None)
parser.add_argument('--n_iters', type=int, default=3100)


def main(args):
    args.save_path = f"{args.ckpt_path}.unroll.pkl"

    env = create_env.create_env(args.env_id)
    rng = jax.random.PRNGKey(0)

    n_envs = 128

    rng, _rng = split(rng)
    env_params = env.sample_params(_rng)

    d_obs, = env.observation_space(env_params).shape
    n_acts = env.action_space(env_params).n

    d_obs_uni = 64
    n_acts_uni = 18
    T = 128

    agent = BCTransformer(n_acts=n_acts_uni, n_layers=4, n_heads=8, d_embd=256, n_steps=T)

    with open(args.dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    obs_mean, obs_std = dataset['obs'].mean(axis=(0, 1)), dataset['obs'].std(axis=(0, 1))

    with open(args.ckpt_path, 'rb') as f:
        agent_params = pickle.load(f)['params']

    rng = jax.random.PRNGKey(0)
    obs_mat = jax.random.orthogonal(rng, n=max(d_obs, d_obs_uni), shape=())[:d_obs_uni, :d_obs]

    env_step = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))
    agent_forward = jax.jit(jax.vmap(agent.apply, in_axes=(None, 0, 0)))

    obs, state = jax.vmap(env.reset, in_axes=(0, None))(split(_rng, n_envs), env_params)
    obs = ((obs - obs_mean) / (obs_std + 1e-5)) @ obs_mat.T

    act0 = jnp.zeros((n_envs,), dtype=int)
    x_obs, x_act = [obs], [act0]
    rews, dones, rets = [], [], []

    pbar = tqdm(range(args.n_iters))
    for t in pbar:

        len_obs = len(x_obs)
        if len_obs < T:
            x_obsv, x_actv = jnp.stack(x_obs + [obs] * (T - len_obs), axis=1), jnp.stack(x_act + [act0] * (T - len_obs),
                                                                                         axis=1)
            logits = agent_forward(agent_params, x_obsv, x_actv)
            rng, _rng = split(rng)
            act = jax.random.categorical(_rng, logits[:, len_obs - 1, :n_acts])
        else:
            x_obsv, x_actv = jnp.stack(x_obs, axis=1), jnp.stack(x_act, axis=1)
            logits = agent_forward(agent_params, x_obsv, x_actv)
            rng, _rng = split(rng)
            act = jax.random.categorical(_rng, logits[:, -1, :n_acts])
            # act = logits[:, time, :2].argmax(axis=-1)

        rng, _rng = split(rng)
        obs, state, rew, done, info = env_step(split(_rng, n_envs), state, act, env_params)
        obs = ((obs - obs_mean) / (obs_std + 1e-5)) @ obs_mat.T
        rets.append(info['returned_episode_returns'])
        rews.append(rew)
        dones.append(done)

        x_obs.append(obs)
        x_act[-1] = act
        x_act.append(act0)
        x_obs, x_act = x_obs[-T:], x_act[-T:]

        pbar.set_postfix(rets=rets[-1].mean())

    rews = jnp.stack(rews, axis=0)
    dones = jnp.stack(dones, axis=0)
    rets = jnp.stack(rets, axis=0)

    rets = np.asarray(rets[-500:, :]).mean(axis=0)
    with open(args.save_path, 'wb') as f:
        pickle.dump(dict(rets=rets), f)
    print(f"Score: {rets.mean()}")


if __name__ == '__main__':
    main(parser.parse_args())
