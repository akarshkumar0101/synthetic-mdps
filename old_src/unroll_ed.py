import argparse
import pickle

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from jax import config as jax_config
from jax.random import split
from tqdm.auto import tqdm

import data_utils
import util
from agents.regular_transformer import BCTransformer

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

    dataset_train, _, transform_params = data_utils.construct_dataset(args.dataset_paths, args.exclude_dataset_paths,
                                                                      args.d_obs_uni, args.d_act_uni,
                                                                      percent_dataset=(args.percent_dataset, 1.0))
    transform_params = jax.tree_map(lambda x: jnp.array(x), transform_params)

    agent = BCTransformer(d_obs=args.d_obs_uni, d_act=args.d_act_uni,
                          n_layers=args.n_layers, n_heads=args.n_heads, d_embd=args.d_embd, ctx_len=args.ctx_len,
                          mask_type=args.mask_type)
    agent_forward = jax.jit(jax.vmap(agent.apply, in_axes=(None, 0, 0, 0)))

    batch = data_utils.sample_batch_from_dataset(rng, dataset_train, args.num_envs, args.ctx_len, args.seq_len)
    print(jax.tree_map(lambda x: x.shape, batch))

    rng, _rng = split(rng)
    agent_params_random = agent.init(_rng, batch['obs'][0], batch['act'][0], batch['time'][0])

    with open(args.load_ckpt, 'rb') as f:
        agent_params = pickle.load(f)['params']

    T = args.ctx_len - 2

    def sample_weird_batch(rng):
        _rng1, _rng2 = split(rng)
        batch1 = data_utils.sample_batch_from_dataset(_rng1, dataset_train, args.num_envs, args.ctx_len, args.seq_len)
        batch2 = data_utils.sample_batch_from_dataset(_rng2, dataset_train, args.num_envs, args.ctx_len, args.seq_len)
        batch = jax.tree_map(lambda x: x.copy(), batch1)
        for key in ['obs', 'act']:
            batch[key][:, T:] = batch2[key][:, T:]
        return batch

    def unroll_agent(agent_params, buffer):
        buffer = jax.tree_map(lambda x: x.copy(), buffer)

        obs, infos = envs.reset()
        stats = []
        for _ in tqdm(range(1000)):
            buffer['obs'][:, T] = data_utils.transform_obs(obs, transform_params[0])
            out = agent_forward(agent_params, buffer['obs'], buffer['act'], buffer['time'])
            act_pred = out['act_now_pred'][:, T - 1]
            act_original = data_utils.inverse_transform_act(act_pred, transform_params[0])
            obs, rew, term, trunc, infos = envs.step(act_original)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        stats.append((info["episode"]["r"], info["episode"]["l"]))
        return np.array(stats)

    def test_loss(agent_params, rng, n_iters=500):
        out = []
        for i_iter in tqdm(range(n_iters)):
            rng, _rng = split(rng)
            batch = sample_weird_batch(_rng)
            outi = agent_forward(agent_params, batch['obs'], batch['act'], batch['time'])
            out.append(outi)
        out = util.tree_cat(out)

        act = out['act_now'][:, :, :]
        act_pred = out['act_now_pred'][:, :, :]

        mse = ((act - act_pred) ** 2).mean(axis=-1).mean(axis=0)
        print(f"Action Loss Universal: ctx_mean: {mse.mean(): .4f}, ctx_T: {mse[T - 1]: .4f}")

        act = data_utils.inverse_transform_act(act, transform_params[0])
        act_pred = data_utils.inverse_transform_act(act_pred, transform_params[0])
        mse = ((act - act_pred) ** 2).mean(axis=-1).mean(axis=0)
        print(f"Action Loss  Original: ctx_mean: {mse.mean(): .4f}, ctx_T: {mse[T - 1]: .4f}")

    print("------------ RANDOM AGENT ------------")
    test_loss(agent_params_random, jax.random.PRNGKey(0))

    # stats = unroll_agent(agent_params_random, buffer1)
    # print(f"Mean Return: {stats[:, 0].mean(): .4f}, Mean Length: {stats[:, 1].mean(): .4f}")

    print("------------ LOADED AGENT ------------")
    test_loss(agent_params, jax.random.PRNGKey(0))

    # stats = unroll_agent(agent_params, buffer1)
    # print(f"Mean Return: {stats[:, 0].mean(): .4f}, Mean Length: {stats[:, 1].mean(): .4f}")


def make_env(env_id, idx, vid_name, gamma):
    def thunk():
        if vid_name is not None and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, vid_name)
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


def do_rollout(agent, agent_params, env_id, dataset_train, dataset_test, transform_params,
               num_envs=8, video_dir=None, seed=0, ctx_len=256, seq_len=256):
    rng = jax.random.PRNGKey(seed)
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, i, False, None, 0.99) for i in range(num_envs)]
    )
    transform_params = jax.tree_map(lambda x: jnp.array(x), transform_params)

    agent_forward = jax.jit(jax.vmap(agent.apply, in_axes=(None, 0, 0, 0)))

    T = ctx_len - 2

    def sample_weird_batch(rng):
        _rng1, _rng2 = split(rng)
        batch1 = data_utils.sample_batch_from_dataset(_rng1, dataset_train, num_envs, ctx_len, seq_len)
        batch2 = data_utils.sample_batch_from_dataset(_rng2, dataset_train, num_envs, ctx_len, seq_len)
        batch = jax.tree_map(lambda x: x.copy(), batch1)
        for key in ['obs', 'act']:
            batch[key][:, T:] = batch2[key][:, T:]
        return batch

    def unroll_agent(agent_params, buffer):
        buffer = jax.tree_map(lambda x: x.copy(), buffer)

        obs, infos = envs.reset()
        stats = []
        for _ in tqdm(range(1000), desc="Rollout"):
            buffer['obs'][:, T] = data_utils.transform_obs(obs, transform_params[0])
            out = agent_forward(agent_params, buffer['obs'], buffer['act'], buffer['time'])
            act_pred = out['act_now_pred'][:, T - 1]
            act_original = data_utils.inverse_transform_act(act_pred, transform_params[0])
            obs, rew, term, trunc, infos = envs.step(act_original)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        stats.append((info["episode"]["r"], info["episode"]["l"]))
        return np.array(stats)

    def test_loss(agent_params, batch):
        out = agent_forward(agent_params, batch['obs'], batch['act'], batch['time'])
        mse_act = ((out['act_now'] - out['act_now_pred']) ** 2).mean(axis=-1).mean(axis=0)
        mse_obs = ((out['obs_nxt'] - out['obs_nxt_pred']) ** 2).mean(axis=-1).mean(axis=0)
        return np.array(mse_act), np.array(mse_obs)

    def unroll_agent_multi(agent_params, rng, n_iters):
        stats = []
        for i_iter in range(n_iters):
            rng, _rng = split(rng)
            batch = sample_weird_batch(_rng)
            statsi = unroll_agent(agent_params, batch)
            stats.append(statsi)
        return np.concatenate(stats)

    def test_loss_multi(agent_params, rng, n_iters):
        mse_act, mse_obs = [], []
        for i_iter in tqdm(range(n_iters), desc="Test Loss"):
            rng, _rng = split(rng)
            batch = sample_weird_batch(_rng)
            out = agent_forward(agent_params, batch['obs'], batch['act'], batch['time'])
            mse_acti = ((out['act_now'] - out['act_now_pred']) ** 2).mean(axis=-1).mean(axis=0)
            mse_obsi = ((out['obs_nxt'] - out['obs_nxt_pred']) ** 2).mean(axis=-1).mean(axis=0)
            mse_act.append(mse_acti)
            mse_obs.append(mse_obsi)
        return np.stack(mse_act, axis=0).mean(axis=0), np.stack(mse_obs, axis=0).mean(axis=0)

    mse_act, mse_obs = test_loss_multi(agent_params, jax.random.PRNGKey(0), 100)
    stats = unroll_agent_multi(agent_params, jax.random.PRNGKey(0), 1)
    print("MSE Action: ", mse_act.mean())
    print("MSE Action final: ", mse_act[-1].item())
    print("Rollout score: ", stats[:, 0].mean())
    return mse_act, mse_obs, stats


def rollout_transformer(agent, agent_params, env_id, transform_params, prompt=None,
                        num_envs=8, num_steps=1000, vid_name=None, seed=0):
    rng = jax.random.PRNGKey(seed)
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, i, vid_name, 0.99) for i in range(num_envs)]
    )
    transform_params = jax.tree_map(lambda x: jnp.array(x), transform_params)
    agent_forward = jax.jit(jax.vmap(agent.apply, in_axes=(None, 0, 0, 0)))

    if prompt is None:
        prompt_obs = jnp.zeros((num_envs, 0, d_obs))
        prompt_act = jnp.zeros((num_envs, 0, d_act))
    else:
        prompt_obs, prompt_act = prompt
    prompt_len = prompt_obs.shape[1]

    ctx_len = agent.ctx_len - prompt_len
    
    def get_act_pred(obs_list, act_list):
        obs_v, act_v = jnp.stack(obs_list, axis=1), jnp.stack(act_list, axis=1)
        
        T = obs_v.shape[1]
        if T<ctx_len:
            obs_filler = jnp.zeros((num_envs, ctx_len-T, agent.d_obs))
            act_filler = jnp.zeros((num_envs, ctx_len-T, agent.d_act))
            obs_v = jnp.concatenate([prompt_obs, obs_v, obs_filler], axis=1)
            act_v = jnp.concatenate([prompt_act, act_v, act_filler], axis=1)
        elif T==ctx_len:
            obs_v = jnp.concatenate([prompt_obs, obs_v], axis=1)
            act_v = jnp.concatenate([prompt_act, act_v], axis=1)
        else:
            raise NotImplementedError
        
        out = agent_forward(agent_params, obs_v, act_v, None)
        return out['act_pred'][:, prompt_len+T-1]

    stats = []
    
    obs_list, act_list = [], []

    obs, infos = envs.reset()
    for _ in tqdm(range(num_steps), desc="Rollout"):
        obs_list.append(data_utils.transform_obs(obs, transform_params))
        act_list.append(jnp.zeros((num_envs, agent.d_act)))
        obs_list, act_list = obs_list[-ctx_len:], act_list[-ctx_len:]
        
        act_pred = get_act_pred(obs_list, act_list)
        act_list[-1] = act_pred
        
        act_original = data_utils.inverse_transform_act(act_pred, transform_params)
        obs, rew, term, trunc, infos = envs.step(act_original)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    stats.append((info["episode"]["r"], info["episode"]["l"]))
    return np.array(stats)[:, 0]


if __name__ == '__main__':
    main(parser.parse_args())
