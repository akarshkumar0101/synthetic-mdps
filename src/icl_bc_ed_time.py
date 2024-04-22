import argparse
import os
# print('CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'])
# print('XLA_PYTHON_CLIENT_PREALLOCATE', os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'])
# print('XLA_PYTHON_CLIENT_MEM_FRACTION', os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'])
import pickle

import flax.linen as nn
import jax
import numpy as np
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from jax import config as jax_config
from jax.random import split
from tqdm.auto import tqdm
from einops import rearrange

import data_utils
import unroll_ed
from agents.regular_transformer import BCTransformer
from util import save_pkl, tree_stack

jax_config.update("jax_debug_nans", True)

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
group.add_argument("--n_augs", type=int, default=0)
group.add_argument("--aug_dist", type=str, default="uniform")
group.add_argument("--n_segs", type=int, default=1)
group.add_argument("--nv", type=int, default=4096)
group.add_argument("--nh", type=int, default=131072)
group.add_argument("--seg_diff_envs", type=lambda x: x=='True', default=False)

group = parser.add_argument_group("optimization")
group.add_argument("--n_iters_eval", type=int, default=100)
group.add_argument("--n_iters", type=int, default=100000)
group.add_argument("--bs", type=int, default=64)
# group.add_argument("--mini_bs", type=int, default=None)
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
# group.add_argument("--seq_len", type=int, default=512)  # how long history it can see
group.add_argument("--mask_type", type=str, default="causal")

group = parser.add_argument_group("rollout")
group.add_argument("--env_id", type=str, default=None)
group.add_argument("--n_iters_rollout", type=int, default=1000)
group.add_argument("--video", type=lambda x: x=='True', default=False)


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if v == "None":
            setattr(args, k, None)  # set all "none" to None
    # if args.mini_bs is None:
        # args.mini_bs = args.bs
    # assert args.bs % args.mini_bs == 0
    return args


def main(args):
    print(args)
    rng = jax.random.PRNGKey(args.seed)
    # run = wandb.init(entity=args.entity, project=args.project, name=args.name, config=args)

    dataset_train, dataset_test, transform_params = data_utils.construct_dataset(args.dataset_paths,
                                                                                 args.exclude_dataset_paths,
                                                                                 args.d_obs_uni, args.d_act_uni,
                                                                                 nvh=(args.nv, args.nh))

    print("----------------------------")
    print(f"Train Dataset shape: {jax.tree_map(lambda x: (type(x), x.shape, x.dtype), dataset_train)}")
    print(f"Test Dataset shape: {jax.tree_map(lambda x: (type(x), x.shape, x.dtype), dataset_test)}")

    agent = BCTransformer(d_obs=args.d_obs_uni, d_act=args.d_act_uni,
                          n_layers=args.n_layers, n_heads=args.n_heads, d_embd=args.d_embd, ctx_len=args.ctx_len,
                          mask_type=args.mask_type)

    batch = data_utils.sample_batch_from_dataset(rng, dataset_train, args.bs, args.ctx_len)

    rng, _rng = split(rng)
    agent_params = agent.init(_rng, batch['obs'][0], batch['act'][0], batch['time'][0])

    lr_schedule = optax.constant_schedule(args.lr)
    tx = optax.chain(optax.clip_by_global_norm(args.clip_grad_norm),
                     optax.adamw(lr_schedule, weight_decay=args.weight_decay, eps=1e-8))
    train_state = TrainState.create(apply_fn=agent.apply, params=agent_params, tx=tx)

    def loss_fn(agent_params, batch):
        result = jax.vmap(agent.apply, in_axes=(None, 0, 0, 0))(agent_params, batch['obs'], batch['act'], batch['time'])
        act_pred, obs_pred = result['act_pred'], result['obs_pred']
        mse_act = ((result['act_pred'] - batch['act'])**2).mean(axis=-1) # mean over dim
        mse_act = mse_act.mean(axis=0)  # mean over batch
        mse_obs = jnp.zeros_like(mse_act) # TODO: do obs prediction
        loss = mse_act.mean() # mean over ctx
        metrics = dict(loss=loss, mse_act=mse_act, mse_obs=mse_obs)
        return loss, metrics

    def iter_test(train_state, batch):
        loss, metrics = loss_fn(train_state.params, batch)
        return train_state, metrics

    def iter_train(train_state, batch):
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params, batch)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, metrics

    def sample_batch_segments_from_dataset(rng, dataset, batch_size, n_segs, seg_len):
        rng, _rng1, _rng2 = split(rng, 3)
        n_e, n_t, *_ = dataset['obs'].shape
        if args.seg_diff_envs:
            i_e = jax.random.randint(_rng1, (batch_size, n_segs, 1), minval=0, maxval=n_e)
        else:
            i_e = jax.random.randint(_rng1, (batch_size, 1, 1), minval=0, maxval=n_e)
        i_t = jax.random.randint(_rng2, (batch_size, n_segs, 1), minval=0, maxval=n_t - seg_len)
        i_t = i_t + jnp.arange(seg_len)
        batch = jax.tree_map(lambda x: x[i_e, i_t, ...], dataset)
        return batch

    def sample_test_batch(rng):
        rng, _rng_batch, _rng_aug = split(rng, 3)
        batch = sample_batch_segments_from_dataset(_rng_batch, dataset_test, args.bs, args.n_segs, args.ctx_len//args.n_segs)
        batch = jax.tree_map(lambda x: rearrange(x, 'b s t ... -> b (s t) ...'), batch)
        batch = data_utils.augment_batch(_rng_aug, batch, 0)
        return rng, batch

    def sample_train_batch(rng):
        rng, _rng_batch, _rng_aug = split(rng, 3)
        batch = sample_batch_segments_from_dataset(_rng_batch, dataset_train, args.bs, args.n_segs, args.ctx_len//args.n_segs)
        batch = jax.tree_map(lambda x: rearrange(x, 'b s t ... -> b (s t) ...'), batch)
        batch = data_utils.augment_batch(_rng_aug, batch, args.n_augs, dist=args.aug_dist)
        return rng, batch

    iter_test, iter_train = jax.jit(iter_test), jax.jit(iter_train)
    rng, batch = sample_train_batch(rng)

    batch = jax.tree_map(lambda x: jnp.array(x), batch)
    print('--------------------------------')
    print(jax.tree_map(lambda x: type(x), batch))
    # print('--------------------------------')
    # print(jax.tree_map(lambda x: type(x), train_state))
    # print('--------------------------------')

    def train_step_new(train_state, _):
        train_state, metrics_train_i = iter_train(train_state, batch)
        return train_state, _

    # a, b = 1000, 100
    a, b = 100000, 1
    def train_loop(train_state):
        return jax.lax.scan(train_step_new, train_state, jnp.arange(b))
    train_loop = jax.jit(train_loop)

    # --------------------------- TRAINING ---------------------------
    # for i in tqdm(range(a)):
    #     train_state, _ = train_loop(train_state)

    for i in tqdm(range(1000)):
        for j in range(100):
            train_state, _ = train_loop(train_state)


if __name__ == '__main__':
    main(parse_args())
# TODO: keep it mind that multiple dataset makes it much slower. I think its cause of cat operation
