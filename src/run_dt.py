import argparse

import jax
import jax.numpy as jnp
from jax import config
from jax.random import split
from tqdm.auto import tqdm

from agents.basic import RandomAgent
from agents.linear_transformer import DecisionTransformerAgent
from algos.rcbc import make_rcbc_funcs
from run import create_env

config.update("jax_debug_nans", True)

parser = argparse.ArgumentParser()
# experiment args
parser.add_argument("--n_seeds", type=int, default=1)
parser.add_argument("--env_id", type=str, default="name=cartpole;mrl=1x128")
parser.add_argument("--agent_id", type=str, default="linear_transformer")

parser.add_argument("--run", type=str, default='train')
parser.add_argument("--load_dir", type=str, default=None)
parser.add_argument("--save_dir", type=str, default=None)

# parser.add_argument("--save_fig", type=str, default=None)
parser.add_argument("--n_iters", type=int, default=10)

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


def create_agent(agent_id, n_acts, n_steps):
    assert agent_id in ['linear_transformer']
    agent = DecisionTransformerAgent(n_acts, n_steps=n_steps, n_layers=2, n_heads=4, d_embd=128)
    return agent


def run(args):
    print(args)
    if args.load_dir == "None":
        args.load_dir = None
    assert args.run in ['train', 'eval']
    config = vars(args)

    rng = jax.random.PRNGKey(0)

    env = create_env(args.env_id, args.n_steps)
    n_acts = env.action_space(None).n
    agent = create_agent(args.agent_id, n_acts, args.n_steps)
    agent_collect = RandomAgent(n_acts)

    init_agent_env, dt_step = make_rcbc_funcs(
        agent_collect, agent, env,
        args.n_envs, args.n_steps, args.n_updates, args.n_envs_batch, args.lr, args.clip_grad_norm,
        args.ent_coef, args.gamma
    )
    init_agent_env = jax.vmap(init_agent_env)
    dt_step = jax.jit(jax.vmap(dt_step, in_axes=(0, None)))

    rng, _rng = split(rng)
    carry = init_agent_env(split(_rng, args.n_seeds))
    pbar = tqdm(range(args.n_iters))
    losses = []
    for i_iter in pbar:
        carry, (buffer, loss) = dt_step(carry, None)
        losses.append(loss[1][0].mean(axis=(0, 1, 2)))
        pbar.set_postfix(ppl=jnp.e**losses[-1].mean(), ppl_end=jnp.e**losses[-1][-10:].mean())
    losses = jnp.stack(losses, axis=0)

    import matplotlib.pyplot as plt
    plt.plot(jnp.e**losses[:, :].mean(axis=-1), label='mean')
    plt.plot(jnp.e**losses[:, :10].mean(axis=-1), label='start of context')
    plt.plot(jnp.e**losses[:, -10:].mean(axis=-1), label='end of context')
    plt.ylim(1.0, 2.1)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run(parser.parse_args())
