import argparse
import pickle

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from jax import config as jax_config
from jax.random import split
from tqdm.auto import tqdm

from agents.regular_transformer import BCTransformer
from icl_bc_ed import construct_dataset, sample_batch_from_dataset
from util import save_pkl

jax_config.update("jax_debug_nans", True)

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=0)
parser.add_argument('--load_ckpt', type=str, default=None)
parser.add_argument('--save_dir', type=str, default=None)

group = parser.add_argument_group("data")
group.add_argument("--dataset_paths", type=str, nargs="+", default=[])
group.add_argument("--exclude_dataset_paths", type=str, nargs="+", default=[])
group.add_argument("--percent_dataset", type=float, default=1.0)
# group.add_argument("--n_augs_test", type=int, default=0)
group.add_argument("--n_augs", type=int, default=0)
group.add_argument("--aug_dist", type=str, default="uniform")

group = parser.add_argument_group("unroll")
group.add_argument('--env_id', type=str, default=None)
group.add_argument('--num_envs', type=int, default=8)
group.add_argument('--n_iters', type=int, default=3100)

# Model arguments
group = parser.add_argument_group("model")
group.add_argument("--d_obs_uni", type=int, default=128)
group.add_argument("--d_act_uni", type=int, default=21)
group.add_argument("--n_layers", type=int, default=4)
group.add_argument("--n_heads", type=int, default=8)
group.add_argument("--d_embd", type=int, default=256)
group.add_argument("--ctx_len", type=int, default=512)  # physical ctx_len of transformer
group.add_argument("--seq_len", type=int, default=512)  # how long history it can see
group.add_argument("--mask_type", type=str, default="causal")


def main(args):
    rng = jax.random.PRNGKey(args.seed)

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, False, None, 0.99) for i in range(args.num_envs)]
    )
    d_obs = envs.observation_space.shape[-1]
    d_act = envs.action_space.shape[-1]
    d_obs_uni, d_act_uni = args.d_obs_uni, args.d_act_uni

    dataset_train, dataset_test, (obs_mean, obs_std, act_mean, act_std) = construct_dataset(args.dataset_paths,
                                                                                            args.exclude_dataset_paths,
                                                                                            args.d_obs_uni,
                                                                                            args.d_act_uni,
                                                                                            percent_dataset=(
                                                                                                args.percent_dataset,
                                                                                                1.0))

    print('obs_mean', obs_mean.shape)
    print('obs_std', obs_std.shape)
    print('act_mean', act_mean.shape)
    print('act_std', act_std.shape)

    agent = BCTransformer(d_obs=args.d_obs_uni, d_act=args.d_act_uni,
                          n_layers=args.n_layers, n_heads=args.n_heads, d_embd=args.d_embd, ctx_len=args.ctx_len,
                          mask_type=args.mask_type)
    with open(args.load_ckpt, 'rb') as f:
        agent_params = pickle.load(f)['params']
    print(jax.tree_map(lambda x: x.shape, agent_params))

    rng = jax.random.PRNGKey(0)
    obs_mat = jax.random.orthogonal(rng, max(d_obs, d_obs_uni))[:d_obs_uni, :d_obs]
    act_mat = jax.random.orthogonal(rng, max(d_act, d_act_uni))[:d_act_uni, :d_act]
    obs_mat = obs_mat / np.sqrt(np.diag(obs_mat @ obs_mat.T))[:, None]  # to make sure output is standard normal
    act_mat = act_mat / np.sqrt(np.diag(act_mat @ act_mat.T))[:, None]
    obs_mat, act_mat = np.array(obs_mat, dtype=np.float32), np.array(act_mat, dtype=np.float32)

    def transform_obs(obs):
        obs = (obs - obs_mean) / (obs_std + 1e-8)
        return obs @ obs_mat.T

    def transform_act(act):
        act = (act - act_mean) / (act_std + 1e-8)
        return act @ act_mat.T

    def inverse_transform_act(act):
        act = act @ act_mat
        act = act * act_std + act_mean
        return act

    obs, info = envs.reset()
    rews = []
    for i in tqdm(range(1000)):
        act = envs.action_space.sample()
        act = act + np.random.normal(size=act.shape)
        obs, rew, term, trunc, info = envs.step(act)
        rews.append(rew)
    rews = np.stack(rews, axis=-1)
    print("Random policy: ")
    print(rews.sum(axis=-1).mean())

    buffer = sample_batch_from_dataset(rng, dataset_train, args.num_envs, 512, 512)
    print(jax.tree_map(lambda x: x.shape, buffer))

    agent_forward = jax.jit(jax.vmap(agent.apply, in_axes=(None, 0, 0, 0)))
    out = agent_forward(agent_params, buffer['obs'], buffer['act'], buffer['time'])
    print(jax.tree_map(lambda x: x.shape, out))

    print("Action Loss: ")
    mse = ((out['act_now'] - out['act_now_pred']) ** 2).mean(axis=(0, -1))
    print(f"{mse[0]=}, {mse[-1]=}, {mse.mean()=}")
    print("Observation Loss: ")
    mse = ((out['obs_nxt'] - out['obs_nxt_pred']) ** 2).mean(axis=(0, -1))
    print(f"{mse[0]=}, {mse[-1]=}, {mse.mean()=}")

    t = 505
    print('heheheh--------------------------------------------------------------------------------')

    obs, info = envs.reset()
    buffer['obs'][:, t] = transform_obs(obs)
    rews = []
    for i in tqdm(range(1000)):
        out = agent_forward(agent_params, buffer['obs'], buffer['act'], buffer['time'])
        act = out['act_now_pred'][:, t - 1]

        envact = inverse_transform_act(act)
        envact = envact + np.random.normal(size=envact.shape)
        obs, rew, term, trunc, info = envs.step(envact)

        buffer['act'][:, t] = act

        # t = t + 1
        buffer['obs'][:, t] = transform_obs(obs)

        rews.append(rew)
    rews = np.stack(rews, axis=-1)
    print("Agent policy: ")
    print(rews.sum(axis=-1).mean())


    # rng = jax.random.PRNGKey(1000)
    # buffer2 = sample_batch_from_dataset(rng, dataset_test, args.num_envs, 512, 512)
    #
    # my_act, optimal_act = [], []
    # obs = buffer2['obs'][:, 0]
    # buffer['obs'][:, t] = obs
    # rews = []
    # for i in tqdm(range(255)):
    #     out = agent_forward(agent_params, buffer['obs'], buffer['act'], buffer['time'])
    #     act = out['act_now_pred'][:, t - 1]
    #
    #     my_act.append(act)
    #     optimal_act.append(buffer2['act'][:, i])
    #
    #     # obs, rew, term, trunc, info = envs.step(inverse_transform_act(act))
    #     obs = buffer2['obs'][:, i+1]
    #
    #     buffer['act'][:, t] = act
    #
    #     t = t + 1
    #     buffer['obs'][:, t] = obs
    #
    # my_act = np.stack(my_act, axis=1)
    # optimal_act = np.stack(optimal_act, axis=1)
    # print(my_act.shape, optimal_act.shape)
    # print(((my_act - optimal_act) ** 2).mean())
    # print(((my_act - optimal_act) ** 2).mean(axis=(0, -1)))




    return

    obs_mean, obs_std = dataset['obs'].mean(axis=(0, 1)), dataset['obs'].std(axis=(0, 1))
    expert_batch = sample_batch_from_dataset(jax.random.PRNGKey(0), dataset, args.n_envs, args.ctx_len // 2)

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


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


if __name__ == '__main__':
    main(parser.parse_args())
