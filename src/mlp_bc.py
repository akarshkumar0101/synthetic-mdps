import argparse
import os

import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from flax.training.train_state import TrainState
from jax.random import split
from tqdm.auto import tqdm

import data_utils

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--dataset_path", type=str, default=None)
parser.add_argument("--save_dir", type=str, default=None)
parser.add_argument("--env_id", type=str, default="HalfCheetah-v4")

parser.add_argument("--d_obs_uni", type=int, default=64)
parser.add_argument("--d_act_uni", type=int, default=21)

parser.add_argument("--transform", type=str, default="none")


def make_env(env_id, idx, capture_video, vid_name, gamma=0.99):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"/data/vision/phillipi/akumar01/synthetic-mdps-data/videos/{vid_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        return env

    return thunk


class Agent(nn.Module):
    d_obs: int
    d_act: int

    @nn.compact
    def __call__(self, x):
        act = nn.Sequential([
            nn.Dense(64),
            nn.tanh,
            nn.Dense(64),
            nn.tanh,
            nn.Dense(self.d_act, kernel_init=nn.initializers.normal(stddev=0.01)),
        ])(x)
        return act


def rollout(env_id, n_envs, n_steps, agent_forward, vid_name):
    capture_video = vid_name is not None
    envs = gym.vector.SyncVectorEnv(
        [make_env(f"{env_id}", i, capture_video=capture_video, vid_name=vid_name) for i in range(n_envs)])
    stats = []
    obs, info = envs.reset()
    for _ in tqdm(range(n_steps)):
        act = agent_forward(obs)
        obs, rew, term, trunc, infos = envs.step(act)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    stats.append((info["episode"]["r"], info["episode"]["l"]))
    return np.array(stats)[:, 0].mean()


def sample_batch(rng, dataset, batch_size):
    rng, _rng1, _rng2 = split(rng, 3)
    i_b = jax.random.randint(_rng1, (batch_size,), 0, dataset['obs'].shape[0])
    i_t = jax.random.randint(_rng2, (batch_size,), 0, dataset['obs'].shape[1])
    return jax.tree_map(lambda x: x[i_b, i_t], dataset)


def main(args):
    dataset_ = data_utils.load_dataset(args.dataset_path)
    dataset_train, dataset_test = data_utils.train_test_split(dataset_, percent_train=0.8)
    print("Dataset shape: ", jax.tree_map(lambda x: x.shape, dataset_train))
    print("Dataset shape: ", jax.tree_map(lambda x: x.shape, dataset_test))

    if args.transform == "none":
        transform_params = data_utils.get_identity_transform_params(dataset_train)
    else:
        transform_params = data_utils.get_dataset_transform_params(jax.random.PRNGKey(0), dataset_train,
                                                                   d_obs_uni=args.d_obs_uni, d_act_uni=args.d_act_uni)
    transform_params = jax.tree_map(lambda x: jnp.array(x, dtype=jnp.float32), transform_params)

    rng = jax.random.PRNGKey(args.seed)
    batch = sample_batch(rng, dataset_train, 1)
    batch = data_utils.transform_dataset(batch, transform_params)
    d_obs, d_act = batch['obs'].shape[-1], batch['act'].shape[-1]
    agent = Agent(d_obs=d_obs, d_act=d_act)

    rng, _rng = split(rng)
    agent_params = agent.init(_rng, jax.tree_map(lambda x: x[0], batch['obs']))
    agent_forward = jax.vmap(agent.apply, in_axes=(None, 0))

    # --------------------------------------------------
    batch = sample_batch(rng, dataset_train, 1)
    obs, act = batch['obs'], batch['act']
    obs_t = data_utils.transform_obs(obs, transform_params)
    act_t = data_utils.transform_act(act, transform_params)
    obs_tt = data_utils.inverse_transform_obs(obs_t, transform_params)
    act_tt = data_utils.inverse_transform_act(act_t, transform_params)
    print("obs: ", obs[0, :3])
    print("obs_tt: ", obs_tt[0, :3])
    print("act: ", act[0, :3])
    print("act_tt: ", act_tt[0, :3])

    def iter_train(state, batch):
        def loss_fn(params, batch_use):
            act_pred = agent_forward(params, batch_use['obs'])
            return jnp.mean(jnp.square(act_pred - batch_use['act']))

        print("Batch shape: ")
        print(jax.tree_map(lambda x: x.shape, batch))
        batch_transformed = data_utils.transform_dataset(batch, transform_params)
        print("Batch transformed shape: ")
        print(jax.tree_map(lambda x: x.shape, batch_transformed))

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(state.params, batch_transformed)
        state = state.apply_gradients(grads=grad)
        return state, loss

    def iter_test(state, batch):
        def loss_fn(params, batch_use):
            act_pred = agent_forward(params, batch_use['obs'])
            return jnp.mean(jnp.square(act_pred - batch_use['act']))

        batch_transformed = data_utils.transform_dataset(batch, transform_params)

        act_pred = agent_forward(state.params, batch_transformed['obs'])
        loss_universal = jnp.mean(jnp.square(act_pred - batch_transformed['act']))

        act_pred = data_utils.inverse_transform_act(act_pred, transform_params)
        loss_original = jnp.mean(jnp.square(act_pred - batch['act']))
        return state, (loss_universal, loss_original)

    iter_train, iter_test = jax.jit(iter_train), jax.jit(iter_test)

    tx = optax.chain(  # optax.clip_by_global_norm(1.),
        optax.adamw(3e-4, weight_decay=0., eps=1e-8))
    train_state = TrainState.create(apply_fn=agent.apply, params=agent_params, tx=tx)

    pbar = tqdm(range(10000 + 1))
    loss_val_target = 1.

    df = pd.DataFrame(columns=['loss_train', 'loss_test_universal', 'loss_test_original', 'return'])
    for i in pbar:
        do_rollout = i % 1000 == 0
        if not do_rollout and i % 10 == 0 and i > 0:
            loss_val_current = df.loss_train.ewm(alpha=0.01).mean().iloc[-1]
            if loss_val_current < loss_val_target:
                do_rollout = True
                loss_val_target = loss_val_target * (.1 ** (1 / 4))
                loss_val_target = loss_val_target * (.1 ** (1 / 4))

        if do_rollout:
            def my_forward(x):
                x = data_utils.transform_obs(x, transform_params)
                act = agent_forward(train_state.params, x)
                act = data_utils.inverse_transform_act(act, transform_params)
                return act

            ret = rollout(args.env_id, 64, 1005, my_forward, vid_name=None)
            print(f"Rollout Return: {ret:.4f}")
            df.loc[i, 'return'] = ret

        rng, _rng = split(rng)
        batch = sample_batch(_rng, dataset_train, 64)
        train_state, loss_train = iter_train(train_state, batch)

        rng, _rng = split(rng)
        batch = sample_batch(_rng, dataset_test, 64)
        train_state, (loss_test_universal, loss_test_original) = iter_test(train_state, batch)

        pbar.set_postfix(loss_train=loss_train, loss_test=loss_test_original)
        df.loc[i, 'loss_train'] = loss_train
        df.loc[i, 'loss_test_universal'] = loss_test_universal
        df.loc[i, 'loss_test_original'] = loss_test_original

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        # save_pkl(f"{args.save_dir}", "data", dict(losses_train=losses_train, losses_test=losses_test,
        #                                           rets=rets, iters=iters))
        df.to_csv(f"{args.save_dir}/data.csv")


if __name__ == "__main__":
    main(parser.parse_args())
