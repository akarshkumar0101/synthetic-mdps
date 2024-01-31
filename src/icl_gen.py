import argparse
import os
import pickle
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from einops import rearrange
from jax.random import split
from tqdm.auto import tqdm

import create_env
import util
from agents import DenseObsEmbed
from agents.basic import BigBasicAgentSeparate, BasicAgentSeparate
from algos.ppo_single import PPO

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


def main(args):
    print(args)
    env = create_env.create_env(args.env_id)
    n_acts = env.action_space(None).n

    if args.agent_id == 'small':
        ObsEmbed = partial(DenseObsEmbed, d_embd=32)
        agent = BasicAgentSeparate(ObsEmbed, n_acts)
    elif args.agent_id == 'classic':
        ObsEmbed = partial(DenseObsEmbed, d_embd=128)
        agent = BasicAgentSeparate(ObsEmbed, n_acts)
    elif args.agent_id == 'minatar':
        agent = BigBasicAgentSeparate(n_acts)
    else:
        raise NotImplementedError

    def get_dataset_for_env_params(rng):
        rng, _rng = split(rng)
        env_params = env.sample_params(_rng)

        ppo = PPO(agent, env, env_params,
                  n_envs=args.n_envs, n_steps=args.n_steps, n_updates=args.n_updates, n_envs_batch=args.n_envs_batch,
                  lr=args.lr, clip_grad_norm=args.clip_grad_norm, clip_eps=args.clip_eps,
                  vf_coef=args.vf_coef, ent_coef=args.ent_coef, gamma=args.gamma, gae_lambda=args.gae_lambda)

        init_agent_env_vmap = jax.vmap(ppo.init_agent_env)
        ppo_step_vmap = jax.vmap(ppo.ppo_step, in_axes=(0, None))

        rng, _rng = split(rng)
        carry = init_agent_env_vmap(split(_rng, args.best_of_n_experts))
        carry, buffer = jax.lax.scan(ppo_step_vmap, carry, xs=None, length=args.n_iters_train)
        rets_train = buffer['info']['returned_episode_returns']  # N S T E
        rets_train = rets_train.mean(axis=(-1, -2))
        idx_best_expert = rets_train[-1].argmax()
        rets_train = rets_train[:, idx_best_expert]
        # print(rets.shape, idx_best_expert)
        carry = jax.tree_map(lambda x: x[idx_best_expert], carry)

        carry, buffer = jax.lax.scan(ppo.eval_step, carry, xs=None, length=args.n_iters_eval)
        rets_eval = buffer['info']['returned_episode_returns'].mean()
        dataset = {k: buffer[k] for k in ['obs', 'logits', 'act']}
        # print(jax.tree_map(lambda x: x.shape, dataset))
        dataset = jax.tree_map(lambda x: rearrange(x, 'N T E ... -> (N E) T ...'), dataset)
        # print(jax.tree_map(lambda x: x.shape, dataset))
        # print(rets_eval.shape)
        return dataset, rets_train, rets_eval

    rng = jax.random.PRNGKey(0)
    data = []
    for _ in tqdm(range(args.n_seeds_seq)):
        rng, _rng = split(rng)
        dataset, rets_train, rets_eval = jax.jit(jax.vmap(get_dataset_for_env_params))(split(rng, args.n_seeds_par))
        data.append((dataset, rets_train, rets_eval))
    data = util.tree_stack(data)
    data = jax.tree_map(lambda x: rearrange(x, "S1 S2 ... -> (S1 S2) ..."), data)
    dataset, rets_train, rets_eval = data  # (S E T ...), (S N), (S )
    # print(jax.tree_map(lambda x: x.shape, (dataset, rets_train, rets_eval)))
    dataset = jax.tree_map(lambda x: rearrange(x, 'S E T ... -> (S E) T ...'), dataset)
    print("Dataset shape: ", jax.tree_map(lambda x: x.shape, (dataset, rets_train, rets_eval)))

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)

        n_seeds = args.n_seeds_seq * args.n_seeds_par
        plt.figure(figsize=(10, 5))
        plt.plot(jnp.arange(args.n_iters_train) * (args.n_envs * args.n_steps), rets_train.mean(axis=0), label='mean',
                 color=[1, 0, 0, 1])
        plt.fill_between(jnp.arange(args.n_iters_train) * (args.n_envs * args.n_steps),
                         rets_train.mean(axis=0) - rets_train.std(axis=0) / jnp.sqrt(n_seeds),
                         rets_train.mean(axis=0) + rets_train.std(axis=0) / jnp.sqrt(n_seeds),
                         color=[1, 0, 0, 0.2], label='std error')

        plt.title(f'Training Curve for \n {args.env_id} \n({n_seeds} seeds, best of {args.best_of_n_experts} experts)')
        plt.ylabel('Return')
        plt.xlabel('Env Steps (Training)')
        plt.text(0.5, -0.2, f'Final eval return mean: {rets_eval.mean():6.3f}', fontsize=30, color='r',
                 transform=plt.gca().transAxes, horizontalalignment='center', verticalalignment='center')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{args.save_dir}/training_curve.png', dpi=300)

        with open(f'{args.save_dir}/dataset.pkl', 'wb') as f:
            pickle.dump(dataset, f)
        with open(f'{args.save_dir}/rets_train.pkl', 'wb') as f:
            pickle.dump(rets_train, f)
        with open(f'{args.save_dir}/rets_eval.pkl', 'wb') as f:
            pickle.dump(rets_eval, f)

        try:
            fig = create_env.plot_env_id_multi(args.env_id)
            fig.savefig(f"{args.save_dir}/{args.env_id}.png", dpi='figure')
        except NotImplementedError:
            print(f"Cannot visualize {args.env_id} because it is not implemented")
        plt.close()


if __name__ == '__main__':
    main(parser.parse_args())
