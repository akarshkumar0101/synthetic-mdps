import argparse
import pickle
from functools import partial

import gymnax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from einops import rearrange
from jax.random import split
from tqdm.auto import tqdm

import run
import util
from agents.basic import BasicAgentSeparate
from agents.util import DenseObsEmbed
from algos.ppo_dr import PPO
from mdps.wrappers import LogWrapper
from util import tree_stack

parser = argparse.ArgumentParser()
parser.add_argument("--env_id", type=str, default="CartPole-v1")

parser.add_argument("--load_dir", type=str, default=None)
parser.add_argument("--save_dir", type=str, default=None)

parser.add_argument("--n_tasks", type=int, default=int(1e9))
parser.add_argument("--n_iters", type=int, default=10000)
parser.add_argument("--curriculum", type=str, default="none")

parser.add_argument("--bs", type=int, default=256)
parser.add_argument("--lr", type=float, default=2.5e-4)
parser.add_argument("--clip_grad_norm", type=float, default=1.)


# class MountainCarDenseRew(MyGymnaxWrapper):
#
#     def reset_env(self, key, params):
#         obs, state = self._env.reset_env(key, params)
#         return obs, state
#
#     def step_env(self, key, state, action, params):
#         obs, state, rew, done, info = self._env.step_env(key, state, action, params)
#         # pos_range = jnp.array([-1.2, 0.6])
#         # vel_range = jnp.array([-0.07, 0.07])
#         r = jnp.array([0.6 - -1.2, 0.07 - -0.07])
#         mid = jnp.array([-jnp.pi / 6, 0.])
#         a = jnp.array([state.position, state.velocity])
#         a = ((a - mid) / r)
#         a = jnp.linalg.norm(a)
#         rew = a
#         return obs, state, rew, done, info

def generate_real_env_dataset(env_id):
    rng = jax.random.PRNGKey(0)
    env, env_params = gymnax.make(env_id)
    env.sample_params = lambda rng: env_params
    env = LogWrapper(env)
    n_acts = env.action_space(env_params).n

    ObsEmbed = partial(DenseObsEmbed, d_embd=128)
    agent = BasicAgentSeparate(ObsEmbed, n_acts)

    ppo = PPO(agent, env, sample_env_params=env.sample_params,
              n_envs=4, n_steps=128, n_updates=16, n_envs_batch=1, lr=2.5e-4, clip_grad_norm=0.5,
              clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, gamma=0.99, gae_lambda=0.95)
    init_agent_env = jax.jit(jax.vmap(ppo.init_agent_env))
    ppo_step = jax.jit(jax.vmap(ppo.ppo_step, in_axes=(0, None)))
    eval_step = jax.jit(jax.vmap(ppo.eval_step, in_axes=(0, None)))

    rng, _rng = split(rng)
    carry = init_agent_env(split(_rng, 1))

    rets = []
    for i_iter in tqdm(range(100)):
        carry, buffer = eval_step(carry, None)
        # rets.append(buffer['info']['returned_episode_returns'].mean(axis=(-1, -2)))

    for i_iter in tqdm(range(900)):
        carry, buffer = ppo_step(carry, None)
        rets.append(buffer['info']['returned_episode_returns'].mean(axis=(-1, -2)))

    eval_stuff = []
    for i_iter in tqdm(range(256 * 256)):
        carry, buffer = eval_step(carry, None)
        # rets.append(buffer['info']['returned_episode_returns'].mean(axis=(-1, -2)))

        ks = ['obs', 'logits', 'act', 'rew', 'done']
        eval_stuff.append({k: buffer[k] for k in ks})
    rets = rearrange(jnp.stack(rets), 'N S -> S N')
    # print(jax.tree_map(lambda x: x.shape, buffer))

    eval_stuff = tree_stack(eval_stuff)
    print(jax.tree_map(lambda x: x.shape, eval_stuff))
    eval_stuff = jax.tree_map(lambda x: rearrange(x, 'N 1 T E ... -> (N E) T ...'), eval_stuff)  # only 0th seed
    print(jax.tree_map(lambda x: x.shape, eval_stuff))

    with open(f'../data/temp/expert_data_{env_id}.pkl', 'wb') as f:
        pickle.dump(eval_stuff, f)

    plt.plot(rets.T, c=[0.1, 0.1, 0.1, 0.1])
    plt.plot(rets.mean(axis=0), label='mean')
    plt.legend()
    plt.ylabel('Return')
    plt.xlabel('Training Iteration')

    plt.show()


def generate_syn_env_dataset():
    env_id = "name=csmdp;d_state=2;d_obs=4;n_acts=4;delta=T;trans=linear;rew=goal;tl=64"

    env = run.create_env(env_id)
    ObsEmbed = partial(DenseObsEmbed, d_embd=32)
    agent = BasicAgentSeparate(ObsEmbed, 4)

    def get_dataset(rng):
        rng, _rng = split(rng)
        env_params = env.sample_params(_rng)

        ppo = PPO(agent, env, sample_env_params=lambda rng: env_params,
                  n_envs=4, n_steps=128, n_updates=16, n_envs_batch=1, lr=2.5e-4, clip_grad_norm=0.5,
                  clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, gamma=0.99, gae_lambda=0.95)

        rng, _rng = split(rng)
        carry = ppo.init_agent_env(_rng)

        carry, buffer = jax.lax.scan(ppo.ppo_step, carry, xs=None, length=100)
        rets = buffer['info']['returned_episode_returns']

        carry, buffer = jax.lax.scan(ppo.eval_step, carry, xs=None, length=1)
        buffer = jax.tree_map(lambda x: rearrange(x, 'N T E ... -> (N E) T ...'), buffer)
        dataset = {k: buffer[k] for k in ['obs', 'logits', 'act']}
        return rets, dataset

    rng = jax.random.PRNGKey(0)
    rets, dataset = [], []
    for _ in tqdm(range(6)):
        rng, _rng = split(rng)
        retsi, dataseti = jax.jit(jax.vmap(get_dataset))(split(rng, 64))
        rets.append(retsi)
        dataset.append(dataseti)
    rets = util.tree_stack(rets)
    dataset = util.tree_stack(dataset)
    dataset = jax.tree_map(lambda x: rearrange(x, 'A B C T ... -> (A B C) T ...'), dataset)

    print(rets.shape)
    plt.plot(rets.mean(axis=(0, 1, -1, -2)))
    plt.ylabel('Return')
    plt.xlabel('Training Iteration')
    plt.show()

    print(jax.tree_map(lambda x: x.shape, dataset))
    with open(f'../data/temp/expert_data_{"synthetic"}.pkl', 'wb') as f:
        pickle.dump(dataset, f)
