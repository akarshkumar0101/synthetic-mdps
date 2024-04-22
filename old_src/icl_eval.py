import argparse
import jax.numpy as jnp
import pickle

import envpool
import jax
import numpy as np
from jax import config as jax_config
from jax.random import split
from tqdm.auto import tqdm

from agents.regular_transformer import BCTransformer

jax_config.update("jax_debug_nans", True)

parser = argparse.ArgumentParser()

parser.add_argument("--env_id", type=str, default="name=CartPole-v1")
parser.add_argument("--load_ckpt", type=str, default=None)
# parser.add_argument("--save_dir", type=str, default=None)

parser.add_argument("--n_iters", type=int, default=100)

parser.add_argument("--n_envs", type=int, default=256)
parser.add_argument("--n_steps", type=int, default=128)


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    # set all "none" to None
    for k, v in vars(args).items():
        if v == "None":
            setattr(args, k, None)
    return args


def main(args):
    print(args)
    d_obs_uni = 64
    n_acts_uni = 18
    T = 128

    rng = jax.random.PRNGKey(0)
    agent = BCTransformer(n_acts=n_acts_uni, n_layers=4, n_heads=8, d_embd=256, n_steps=T)
    rng, _rng = split(rng)

    with open(args.load_ckpt, 'rb') as f:
        ckpt = pickle.load(f)
        agent_params = ckpt['params']

    env = envpool.make_gym(task_id='CartPole-v1', num_envs=args.n_envs)
    print(env)
    print(env.observation_space)
    print(env.action_space)

    d_obs, = env.observation_space.shape

    running_ret = np.zeros((args.n_envs,))
    past_ret = np.zeros((args.n_envs,))
    rets = []

    rng, _rng = split(rng)
    W = jax.random.normal(_rng, (d_obs_uni, d_obs))

    obs, info = env.reset()
    past_obs, past_act = [obs], [jnp.zeros((args.n_envs,), dtype=int)]
    for t in tqdm(range(10)):
        past_obs, past_act = past_obs[-T:], past_act[-T:]
        x_obs, x_act = jnp.stack(past_obs, axis=1), jnp.stack(past_act, axis=1)
        # rng, _rng = split(rng)
        print(x_obs.shape, x_act.shape)
        x_obs = x_obs @ W.T
        print(x_obs.shape, x_act.shape)
        out = jax.vmap(agent.apply, in_axes=(None, 0, 0))(agent_params, x_obs, x_act)
        print(out.shape)

        # act = jax.random.randint(_rng, (args.n_envs,), 0, env.action_space.n)
        act = np.random.randint(0, env.action_space.n, (args.n_envs,))
        obs, rew, term, trunc, info = env.step(np.asarray(act))
        done = np.logical_or(term, trunc)
        past_obs.append(obs)
        past_act.append(act)

        past_ret = np.where(done, running_ret, past_ret)
        running_ret += rew
        running_ret *= 1 - done
        rets.append(past_ret)
    rets = np.stack(rets, axis=0)

    print(rets.shape)
    print(rets.mean())


if __name__ == '__main__':
    main(parse_args())
