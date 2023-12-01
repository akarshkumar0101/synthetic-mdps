import argparse
import pickle
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from src.algos import make_train

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


def main(args):
    print(args)
    config = vars(args)
    config = {k.upper(): v for k, v in config.items()}
    config = {k.replace('N_', 'NUM_'): v for k, v in config.items()}
    config['ANNEAL_LR'] = False

    rng = jax.random.PRNGKey(0)
    n_seeds = 6

    n_steps = args.n_steps
    d_obs = 64

    env = create_env(args.env)
    agent = create_agent(env, args.agent, n_steps)

    # -------------------- PRETRAINING --------------------
    mds = jnp.arange(n_seeds)
    rng, _rng = jax.random.split(rng)
    rngs = jax.random.split(_rng, n_seeds)
    rng, _rng = jax.random.split(rng)
    _rng = jax.random.split(_rng, n_seeds)
    init_obs, init_act, init_rew = jnp.zeros((n_steps, d_obs)), jnp.zeros((n_steps,), dtype=jnp.int32), jnp.zeros(
        (n_steps,))
    agent_params = jax.vmap(partial(agent.init, method=agent.forward_parallel),
                            in_axes=(0, None, None, None))(_rng, init_obs, init_act, init_rew)

    pbar = tqdm(total=args.total_timesteps)

    def callback(md, i_iter, traj_batch, pbar=None):
        if md == 0 and pbar is not None:
            pbar.update(args.n_envs * args.n_steps)

    train_fn = make_train(config, env, agent, callback=partial(callback, pbar=pbar), reset_env_iter=True)
    # train_fn = jax.jit(jax.vmap(train_fn))
    train_fn = jax.pmap(train_fn)
    out = train_fn(mds, rngs, agent_params)
    agent_params_trained = out['runner_state'][1].params
    pbar.close()
    rews = out['metrics']  # (n_seeds, n_iters, n_steps)
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
        if args.save_fig == 'show':
            plt.show()
        else:
            plt.savefig(args.save_fig)

    if args.save_agent is not None:
        with open(args.save_agent, 'wb') as f:
            pickle.dump(agent_params_trained, f)


if __name__ == '__main__':
    main(parser.parse_args())


