import os
import sys

import os, sys, glob, pickle
from functools import partial  

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from einops import rearrange, reduce, repeat

import jax
import jax.numpy as jnp
from jax.random import split

import flax.linen as nn
from flax.training.train_state import TrainState

import optax

import gymnasium as gym
import util

import data_utils


def sample_batch(rng, dataset, bs, ctx_len):
    _rng1, _rng2 = split(rng)
    B, T, _ = dataset['obs'].shape
    i_b = jax.random.randint(_rng1, (bs, 1), 0, B)
    i_t = jax.random.randint(_rng2, (bs, 1), 0, T - ctx_len)
    i_t = i_t + jnp.arange(ctx_len)
    return jax.tree_map(lambda x: x[i_b, i_t], dataset)


def augment_batch(rng, batch, n_augs, dist="uniform", mat_type='gaussian'):
    if n_augs == 0:
        return batch
    bs, _, d_obs = batch['obs'].shape
    bs, _, d_act = batch['act'].shape

    def augment_instance(instance, aug_id):
        rng = jax.random.PRNGKey(aug_id)
        _rng_obs, _rng_act = split(rng, 2)
        if mat_type == 'gaussian':
            obs_mat = jax.random.normal(_rng_obs, (d_obs, d_obs)) / jnp.sqrt(d_obs)
            act_mat = jax.random.normal(_rng_act, (d_act, d_act)) / jnp.sqrt(d_act)
        elif mat_type == 'orthogonal':
            obs_mat = jax.random.orthogonal(_rng_obs, d_obs)
            act_mat = jax.random.orthogonal(_rng_act, d_act)
        else:
            raise NotImplementedError
        return dict(obs=instance['obs'] @ obs_mat.T, act=instance['act'] @ act_mat.T)

    if dist == "uniform":
        aug_ids = jax.random.randint(rng, (bs,), minval=0, maxval=n_augs)
    elif dist == "zipf":
        p = 1 / jnp.arange(1, n_augs + 1)
        aug_ids = jax.random.choice(rng, jnp.arange(n_augs), (bs,), p=p / p.sum())
    else:
        raise NotImplementedError

    return jax.vmap(augment_instance)(batch, aug_ids)


from agents.regular_transformer import Block

class BCTransformer(nn.Module):
    d_obs: int
    d_act: int
    n_layers: int
    n_heads: int
    d_embd: int
    ctx_len: int

    mask_type: str = "causal"

    def setup(self):
        self.embed_obs = nn.Dense(features=self.d_embd)
        self.embed_act = nn.Dense(features=self.d_embd)
        self.embed_pos = nn.Embed(num_embeddings=self.ctx_len, features=self.d_embd)

        self.blocks = [Block(n_heads=self.n_heads, mask_type=self.mask_type) for _ in range(self.n_layers)]
        self.ln = nn.LayerNorm()

        self.actor = nn.Dense(features=self.d_act, kernel_init=nn.initializers.orthogonal(0.01))  # T, Da

    def __call__(self, obs, act):  # obs: (T, Do), # act: (T, Da), # time: (T, )
        assert obs.shape[0] == act.shape[0]
        assert obs.shape[0] <= self.ctx_len
        T, _ = obs.shape

        x_obs = self.embed_obs(obs)  # (T, D)
        x_act = self.embed_act(act)  # (T, D)
        x_pos = self.embed_pos(jnp.arange(T))  # (T, D)
        x_act_p = jnp.concatenate([jnp.zeros((1, self.d_embd)), x_act[:-1]], axis=0)

        x = x_obs + x_act_p + x_pos
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)

        act_now_pred = self.actor(x)
        return act_now_pred
    

parser = argparse.ArgumentParser()
# parser.add_argument("--entity", type=str, default=None)
# parser.add_argument("--project", type=str, default="synthetic-mdps")
# parser.add_argument("--name", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--load_ckpt", type=str, default=None)
parser.add_argument("--save_dir", type=str, default=None)
parser.add_argument("--n_ckpts", type=int, default=0)
parser.add_argument("--obj", type=str, default="bc")  # bc or wm

group = parser.add_argument_group("data")
group.add_argument("--dataset_paths", type=str, nargs="+", default=[])
group.add_argument("--exclude_dataset_paths", type=str, nargs="+", default=[])
# group.add_argument("--percent_dataset", type=float, default=1.0)
# group.add_argument("--n_augs_test", type=int, default=0)
group.add_argument("--n_augs", type=int, default=0)
group.add_argument("--aug_dist", type=str, default="uniform")
# group.add_argument("--time_perm", type=lambda x: x == "True", default=False)
# group.add_argument("--zipf", type=lambda x: x == "True", default=False)
group.add_argument("--nv", type=int, default=4096)
group.add_argument("--nh", type=int, default=131072)

group = parser.add_argument_group("optimization")
group.add_argument("--n_iters_eval", type=int, default=10)
group.add_argument("--n_iters", type=int, default=100000)
group.add_argument("--bs", type=int, default=64)
group.add_argument("--mini_bs", type=int, default=None)
group.add_argument("--lr", type=float, default=3e-4)
group.add_argument("--lr_schedule", type=str, default="constant")  # constant or cosine_decay
group.add_argument("--weight_decay", type=float, default=0.)
group.add_argument("--clip_grad_norm", type=float, default=1.)

group = parser.add_argument_group("model")
group.add_argument("--d_obs_uni", type=int, default=128)
group.add_argument("--d_act_uni", type=int, default=21)
group.add_argument("--n_layers", type=int, default=4)
group.add_argument("--n_heads", type=int, default=8)
group.add_argument("--d_embd", type=int, default=256)
group.add_argument("--ctx_len", type=int, default=512)  # physical ctx_len of transformer
group.add_argument("--seq_len", type=int, default=512)  # how long history it can see
group.add_argument("--mask_type", type=str, default="causal")

group = parser.add_argument_group("rollout")
group.add_argument("--env_id", type=str, default=None)
group.add_argument("--n_iters_rollout", type=int, default=100)



data_dir = "~/synthetic-mdps-data/"
data_dir = os.path.expanduser(data_dir)

d_obs_uni, d_act_uni = 128, 16
env_id = "HalfCheetah"
env_id_gym = f"{env_id}-v4"


dataset = data_utils.load_dataset(f"{data_dir}/datasets/mujoco/{env_id}/dataset.pkl")
dataset_train_ori, dataset_test_ori = data_utils.train_test_split(dataset, percent_train=0.8)
del dataset
transform_params = data_utils.get_dataset_transform_params(jax.random.PRNGKey(0), dataset_train_ori,
                                                           d_obs_uni=d_obs_uni, d_act_uni=d_act_uni)

dataset_train_ori = jax.tree_map(lambda x: jnp.array(x), dataset_train_ori)
dataset_test_ori = jax.tree_map(lambda x: jnp.array(x), dataset_test_ori)
transform_params = jax.tree_map(lambda x: jnp.array(x), transform_params)

dataset_train_uni = data_utils.transform_dataset(dataset_train_ori, transform_params)
dataset_test_uni = data_utils.transform_dataset(dataset_test_ori, transform_params)
print(jax.tree_map(lambda x: (x.shape, x.dtype, type(x)), dataset_train_uni))



if __name__=="__main__":
    bs, ctx_len = 32, 128
    agent = BCTransformer(d_obs=d_obs_uni, d_act=d_act_uni, n_layers=2, n_heads=4, d_embd=128*4, ctx_len=ctx_len, mask_type='causal')
    jit_vmap_agent_apply = jax.jit(jax.vmap(agent.apply, in_axes=(None, 0, 0)))

    def train_step(rng, train_state):
        def loss_fn(params, batch):
            out = jax.vmap(agent.apply, in_axes=(None, 0, 0))(params, batch['obs'], batch['act'])
            mse = jnp.mean(jnp.square(out - batch['act']), axis=(0, -1)) # mean over batch, dim
            loss = mse.mean() # mean over ctx
            metrics = dict(loss=loss, mse=mse)
            return loss, metrics
        _rng1, _rng2, _rng3 = split(rng, 3)
        batch = sample_batch(_rng1, dataset_train_uni, bs, ctx_len)
        # batch = augment_batch_permute(_rng3, batch)
        
        # batch = sample_batch(_rng1, dataset_train_uni, bs*ctx_len, 1)
        # batch = jax.tree_map(lambda x: x.reshape(bs, ctx_len, -1), batch)
        
        batch = augment_batch(_rng2, batch, n_augs=int(1e8), dist='uniform', mat_type='orthogonal')
        
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grad = grad_fn(train_state.params, batch)
        train_state = train_state.apply_gradients(grads=grad)
        return train_state, metrics
    train_step = jax.jit(train_step)


    
    rng = jax.random.PRNGKey(0)
    rng, _rng = split(rng)
    batch = sample_batch(_rng, dataset_train_uni, bs, ctx_len)
    agent_params = agent.init(_rng, batch['obs'][0], batch['act'][0])
    print(sum([p.size for p in jax.tree_util.tree_leaves(agent_params)]))

    tx = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(1e-4, weight_decay=0., eps=1e-8))
    train_state = TrainState.create(apply_fn=agent.apply, params=agent_params, tx=tx)

    data = []
    mse = []
    for i in tqdm(range(50000 + 1)):
        rng, _rng = split(rng)
        train_state, metrics = train_step(_rng, train_state)
        mse.append(metrics['mse'])






