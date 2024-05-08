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
from util import save_pkl, save_json, tree_stack

jax_config.update("jax_debug_nans", True)

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
# group.add_argument("--entity", type=str, default=None)
# group.add_argument("--project", type=str, default="synthetic-mdps")
# group.add_argument("--name", type=str, default=None)
group.add_argument("--seed", type=int, default=0)
group.add_argument("--load_ckpt", type=str, default=None)
group.add_argument("--save_dir", type=str, default=None)
group.add_argument("--save_ckpt", type=lambda x: x=='True', default=False)
# group.add_argument("--n_ckpts", type=int, default=0)
# group.add_argument("--obj", type=str, default="bc")  # bc or wm

group = parser.add_argument_group("data")
group.add_argument("--dataset_paths", type=str, default=None)
group.add_argument("--exclude_dataset_paths", type=str, default=None)
group.add_argument("--n_augs", type=int, default=0)
group.add_argument("--aug_dist", type=str, default="uniform")
group.add_argument("--n_segs", type=int, default=1)
group.add_argument("--nv", type=int, default=4096)
group.add_argument("--nh", type=int, default=131072)

group = parser.add_argument_group("optimization")
# group.add_argument("--n_iters_eval", type=int, default=100)
group.add_argument("--n_iters", type=int, default=100000)
group.add_argument("--bs", type=int, default=128)
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
group.add_argument("--n_envs", type=int, default=64)
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

def preprocess_dataset(ds):
    ds_prv = {f"{k}_prv": v[:, :-2] for k, v in ds.items()}
    ds_now = {f"{k}": v[:, 1:-1] for k, v in ds.items()}
    ds_nxt = {f"{k}_nxt": v[:, 2:] for k, v in ds.items()}
    return {**ds_prv, **ds_now, **ds_nxt}

def sample_batch_segments_from_dataset(_rng, dataset, batch_size, n_segs, seg_len):
    _rng1, _rng2 = split(_rng)
    n_e, n_t, *_ = dataset['obs'].shape
    i_e = jax.random.randint(_rng1, (batch_size, 1, 1), minval=0, maxval=n_e)
    i_t = jax.random.randint(_rng2, (batch_size, n_segs, 1), minval=0, maxval=n_t - seg_len)
    i_t = i_t + jnp.arange(seg_len)
    batch = jax.tree_map(lambda x: x[i_e, i_t, ...], dataset)
    batch = jax.tree_map(lambda x: rearrange(x, 'b s t ... -> b (s t) ...'), batch)
    return batch

def main(args):
    print(args)
    save_json(args.save_dir, "config", vars(args))
    
    rng = jax.random.PRNGKey(args.seed)
    # run = wandb.init(entity=args.entity, project=args.project, name=args.name, config=args)

    dataset_train, dataset_test, transform_params = data_utils.construct_dataset(args.dataset_paths,
                                                                                 args.exclude_dataset_paths,
                                                                                 args.d_obs_uni, args.d_act_uni,
                                                                                 nvh=(args.nv, args.nh))
    dataset_train = jax.tree_map(lambda x: jnp.array(x), dataset_train)
    dataset_test = jax.tree_map(lambda x: jnp.array(x), dataset_test)
    transform_params = jax.tree_map(lambda x: jnp.array(x), transform_params)

    print("----------------------------")
    print(f"Train Dataset shape: {jax.tree_map(lambda x: (type(x), x.shape, x.dtype), dataset_train)}")
    print(f"Test Dataset shape: {jax.tree_map(lambda x: (type(x), x.shape, x.dtype), dataset_test)}")

    agent = BCTransformer(d_obs=args.d_obs_uni, d_act=args.d_act_uni,
                          n_layers=args.n_layers, n_heads=args.n_heads, d_embd=args.d_embd, ctx_len=args.ctx_len,
                          mask_type=args.mask_type)

    batch = data_utils.sample_batch_from_dataset(rng, dataset_train, args.bs, args.ctx_len)

    rng, _rng = split(rng)
    if args.load_ckpt is not None:
        with open(args.load_ckpt, "rb") as f:
            agent_params = pickle.load(f)['params']
    else:
        agent_params = agent.init(_rng, batch['obs'][0], batch['act'][0], batch['time'][0])

    print("Agent parameter count: ", sum(p.size for p in jax.tree_util.tree_leaves(agent_params)))
    tabulate_fn = nn.tabulate(agent, jax.random.key(0), compute_flops=True, compute_vjp_flops=True)
    print(tabulate_fn(batch['obs'][0], batch['act'][0], batch['time'][0]))

    if args.lr_schedule == "constant":
        lr_schedule = optax.constant_schedule(args.lr)
    elif args.lr_schedule == "cosine_decay":
        lr_schedule = optax.warmup_cosine_decay_schedule(init_value=0., peak_value=args.lr,
                                                         warmup_steps=args.n_iters//100, decay_steps=args.n_iters-args.n_iters//100,
                                                         end_value=0.0, exponent=1.0)
    else:
        raise NotImplementedError
    tx = optax.chain(optax.clip_by_global_norm(args.clip_grad_norm), optax.adamw(lr_schedule, weight_decay=args.weight_decay, eps=1e-8))
    train_state = TrainState.create(apply_fn=agent.apply, params=agent_params, tx=tx)

    def sample_test_batch(_rng):
        _rng1, _rng2 = split(_rng)
        batch = sample_batch_segments_from_dataset(_rng1, dataset_test, args.bs, args.n_segs, args.ctx_len//args.n_segs)
        batch = data_utils.augment_batch(_rng2, batch, 0)
        return batch

    def sample_train_batch(_rng):
        _rng1, _rng2 = split(_rng)
        batch = sample_batch_segments_from_dataset(_rng1, dataset_train, args.bs, args.n_segs, args.ctx_len//args.n_segs)
        batch = data_utils.augment_batch(_rng2, batch, args.n_augs, dist=args.aug_dist)
        return batch

    def loss_fn(agent_params, batch):
        result = jax.vmap(agent.apply, in_axes=(None, 0, 0, 0))(agent_params, batch['obs'], batch['act'], batch['time'])
        act_pred, obs_pred = result['act_pred'], result['obs_pred']
        mse_act = ((result['act_pred'] - batch['act'])**2).mean(axis=-1) # mean over dim
        mse_act = mse_act.mean(axis=0)  # mean over batch
        mse_obs = ((result['obs_pred'][:, :-1, :] - batch['obs'][:, 1:, :]) ** 2).mean(axis=-1) # mean over dim
        mse_obs = mse_obs.mean(axis=0)  # mean over batch

        loss = mse_act.mean() + mse_obs.mean() # mean over ctx
        metrics = dict(loss=loss, mse_act=mse_act, mse_obs=mse_obs)
        return loss, metrics

    @jax.jit
    def iter_test(train_state, batch):
        _, metrics = loss_fn(train_state.params, batch)
        return train_state, metrics

    @jax.jit
    def iter_train(train_state, batch):
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params, batch)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, metrics

    def iter_test_multiple(train_state, _rng):
        batch = jax.vmap(sample_test_batch)(split(_rng, 100))
        return jax.lax.scan(iter_test, train_state, batch)

    def iter_train_multiple(train_state, _rng):
        batch = jax.vmap(sample_train_batch)(split(_rng, 100))
        return jax.lax.scan(iter_train, train_state, batch)

    # metrics_before, metrics_train, metrics_test, metrics_after = [], [], [], []
    # # --------------------------- BEFORE TRAINING ---------------------------
    # pbar = tqdm(range(args.n_iters_eval), desc="Before Training")
    # for _ in pbar:
    #     rng, batch = sample_test_batch(rng)
    #     train_state, metrics = iter_test(train_state, batch)
    #     pbar.set_postfix(mse_act=metrics['mse_act'].mean().item(), mse_obs=metrics['mse_obs'].mean().item())
    #     metrics_before.append(metrics)
    # save_pkl(args.save_dir, "metrics_before", tree_stack(metrics_before))

    metrics_test, metrics_train, rollout_data = [], [], []
    # --------------------------- TRAINING ---------------------------
    n_iters = args.n_iters // 100
    pbar = tqdm(range(n_iters + 1), desc="Training")
    for i_iter in pbar:
        # if args.save_dir is not None and ((args.n_ckpts > 0 and i_iter == args.n_iters) or (
        #         args.n_ckpts > 1 and i_iter % (args.n_iters // (args.n_ckpts - 1)) == 0)):
        #     save_pkl(args.save_dir, f"ckpt_{i_iter:07d}", dict(i_iter=i_iter, params=train_state.params))
        #     os.system(f"ln -sf \"{args.save_dir}/ckpt_{i_iter:07d}.pkl\" \"{args.save_dir}/ckpt_latest.pkl\"")
        if args.save_ckpt and (i_iter % 100 == 0 or i_iter==n_iters):
            save_pkl(args.save_dir, f"ckpt_latest", dict(i_iter=i_iter, params=train_state.params))

        if args.env_id is not None and (i_iter==0 or i_iter==n_iters):
            num_envs = args.n_envs
            batch = sample_batch_segments_from_dataset(rng, dataset_test, num_envs, args.n_segs-1, args.ctx_len//args.n_segs)
            vid_name = f"{args.save_dir}/videos/{i_iter}" if args.save_dir and args.video else None
            rets, lens, buffer = unroll_ed.rollout_transformer(agent, train_state.params, args.env_id, transform_params[0],
                                                               prompt=(batch['obs'], batch['act']),
                                                               num_envs=num_envs, num_steps=1000, vid_name=vid_name, seed=0)
            print(f"Rollout results: {rets.mean():07.4f} +- {rets.std():07.4f}")
            rollout_data.append(dict(i_iter=i_iter, rets=rets, lens=lens, buffer=buffer, prompt=batch))

        rng, _rng1, _rng2 = split(rng, 3)

        train_state, metrics_test_i = iter_test_multiple(train_state, _rng1)
        train_state, metrics_train_i = iter_train_multiple(train_state, _rng2)

        metrics_test_i = jax.tree_map(lambda x: x.mean(axis=0), metrics_test_i)
        metrics_train_i = jax.tree_map(lambda x: x.mean(axis=0), metrics_train_i)
        pbar.set_postfix(train_loss=metrics_train_i['loss'].mean().item(), test_loss=metrics_test_i['loss'].mean().item())
        metrics_test.append(metrics_test_i)
        metrics_train.append(metrics_train_i)

        if (i_iter % 100 == 0 or i_iter==n_iters):
            save_pkl(args.save_dir, "metrics_train", tree_stack(metrics_train))
            save_pkl(args.save_dir, "metrics_test", tree_stack(metrics_test))
            save_pkl(args.save_dir, "rollout_data", rollout_data)

    # # --------------------------- AFTER TRAINING ---------------------------
    # pbar = tqdm(range(args.n_iters_eval), desc="After Training")
    # for _ in pbar:
    #     rng, batch = sample_test_batch(rng)
    #     train_state, metrics = iter_test(train_state, batch)
    #     pbar.set_postfix(mse_act=metrics['mse_act'].mean().item(), mse_obs=metrics['mse_obs'].mean().item())
    #     metrics_after.append(metrics)
    # save_pkl(args.save_dir, "metrics_after", tree_stack(metrics_after))


if __name__ == '__main__':
    main(parse_args())
# TODO: keep it mind that multiple dataset makes it much slower. I think its cause of cat operation
