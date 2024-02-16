import argparse
import pickle
from functools import partial

import jax
import jax.numpy as jnp
from jax.random import split

from mdps.random_net import RandomMLP, create_random_net

parser = argparse.ArgumentParser()
parser.add_argument("--env_id", type=str, default="CartPole-v1")
parser.add_argument("--agent_id", type=str, default="small")

parser.add_argument("--save_dir", type=str, default=None)

parser.add_argument("--best_of_n_experts", type=int, default=1)
parser.add_argument("--n_seeds_seq", type=int, default=1)  # sequential seeds
parser.add_argument("--n_seeds_par", type=int, default=1)  # parallel seeds

parser.add_argument("--n_iters_train", type=int, default=20)
parser.add_argument("--n_iters_eval", type=int, default=1)  # sequential envs
# ppo args
parser.add_argument("--n_envs", type=int, default=4)  # parallel envs
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


def parse_args(*args, **kwargs):
    return parser.parse_args(*args, **kwargs)


def gen_dataset_zero_act(rng):
    rng, _rng = split(rng)
    obs = jax.random.normal(_rng, (4096, 128, 4))
    logits = jnp.zeros((4096, 128, 1))
    rng, _rng = split(rng)
    act = jax.random.categorical(_rng, logits)

    dataset = dict(obs=obs, logits=logits, act=act)

    print(jax.tree_map(lambda x: x.shape, dataset))

    with open("../data/exp_icl/datasets/synthetic/zero_act/dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)


def gen_dataset_random(rng):
    rng = jax.random.PRNGKey(0)

    net = RandomMLP(n_layers=4, d_hidden=64, d_out=4, activation=jax.nn.relu)

    rng, _rng = split(rng)
    x = jax.random.normal(_rng, (32, 4))
    rng, _rng = split(rng)
    net_params = create_random_net(net, _rng, x)

    rng, _rng = split(rng)
    obs = jax.random.normal(_rng, (4096, 128, 4))
    logits = 5 * jax.vmap(jax.vmap(partial(net.apply, net_params)))(obs)

    rng, _rng = split(rng)
    act = jax.random.categorical(_rng, logits)

    dataset = dict(obs=obs, logits=logits, act=act)
    print(jax.tree_map(lambda x: x.shape, dataset))

    with open("../data/exp_icl/datasets/synthetic/random_function/dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)


def main(args):
    gen_dataset_zero_act(rng=jax.random.PRNGKey(0))
    gen_dataset_random(rng=jax.random.PRNGKey(0))


if __name__ == '__main__':
    main(parser.parse_args())
