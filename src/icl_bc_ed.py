import argparse
import os
# print('CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'])
# print('XLA_PYTHON_CLIENT_PREALLOCATE', os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'])
# print('XLA_PYTHON_CLIENT_MEM_FRACTION', os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'])
import pickle

import flax.linen as nn
import jax
import numpy as np
import optax
from flax.training.train_state import TrainState
from jax import config as jax_config
from jax.random import split
from tqdm.auto import tqdm

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


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if v == "None":
            setattr(args, k, None)  # set all "none" to None
    if args.mini_bs is None:
        args.mini_bs = args.bs
    assert args.bs % args.mini_bs == 0
    return args


def main(args):
    print(args)
    args.n_augs_obs = args.n_augs
    args.n_augs_act = args.n_augs
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

    batch = data_utils.sample_batch_from_dataset(rng, dataset_train, args.bs, args.ctx_len, args.seq_len)

    rng, _rng = split(rng)
    if args.load_ckpt is not None:
        with open(args.load_ckpt, "rb") as f:
            agent_params = pickle.load(f)['params']
    else:
        agent_params = agent.init(_rng, batch['obs'][0], batch['act'][0], batch['time'][0])

    print("Agent parameter count: ", sum(p.size for p in jax.tree_util.tree_leaves(agent_params)))
    tabulate_fn = nn.tabulate(agent, jax.random.key(0), compute_flops=True, compute_vjp_flops=True)
    print(tabulate_fn(batch['obs'][0], batch['act'][0], batch['time'][0]))

    lr_warmup = optax.linear_schedule(0., args.lr, args.n_iters // 100)
    if args.lr_schedule == "constant":
        lr_const = optax.constant_schedule(args.lr)
        lr_schedule = optax.join_schedules([lr_warmup, lr_const], [args.n_iters // 100])
    elif args.lr_schedule == "cosine_decay":
        lr_decay = optax.cosine_decay_schedule(args.lr, args.n_iters - args.n_iters // 100, 1e-1)
        lr_schedule = optax.join_schedules([lr_warmup, lr_decay], [args.n_iters // 100])
    else:
        raise NotImplementedError
    tx = optax.chain(optax.clip_by_global_norm(args.clip_grad_norm),
                     optax.adamw(lr_schedule, weight_decay=args.weight_decay, eps=1e-8))
    train_state = TrainState.create(apply_fn=agent.apply, params=agent_params, tx=tx)

    def loss_fn(agent_params, batch):
        result = jax.vmap(agent.apply, in_axes=(None, 0, 0, 0))(agent_params, batch['obs'], batch['act'], batch['time'])
        act_now, act_now_pred = result['act_now'], result['act_now_pred']
        obs_nxt, obs_nxt_pred = result['obs_nxt'], result['obs_nxt_pred']
        mse_act = ((act_now - act_now_pred) ** 2).mean(axis=-1)
        mse_obs = ((obs_nxt - obs_nxt_pred) ** 2).mean(axis=-1)

        mse_act, mse_obs = mse_act.mean(axis=0), mse_obs.mean(axis=0)  # mean over batch
        loss = mse_act.mean() + mse_obs.mean()  # mean over ctx

        # if len(transform_params) == 1:
        #     act_now = data_utils.inverse_transform_act(act_now, transform_params[0])
        #     act_now_pred = data_utils.inverse_transform_act(act_now_pred, transform_params[0])
        #     obs_nxt = data_utils.inverse_transform_obs(obs_nxt, transform_params[0])
        #     obs_nxt_pred = data_utils.inverse_transform_obs(obs_nxt_pred, transform_params[0])
        #     mse_act = ((act_now - act_now_pred) ** 2).mean(axis=-1)
        #     mse_obs = ((obs_nxt - obs_nxt_pred) ** 2).mean(axis=-1)
        #     mse_act, mse_obs = mse_act.mean(axis=0), mse_obs.mean(axis=0)  # mean over batch

        metrics = dict(loss=loss, mse_act=mse_act, mse_obs=mse_obs)
        return loss, metrics

    def iter_test(train_state, batch):
        loss, metrics = loss_fn(train_state.params, batch)
        return train_state, metrics

    def iter_train(train_state, batch):
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params, batch)
        # grads_zero = jax.tree_map(lambda x: jnp.zeros_like(x), grads)
        # grads_nz = grads
        # grads = grads_zero
        # grads['params']['actor']['kernel'] = grads_nz['params']['actor']['kernel']
        # grads['params']['actor']['bias'] = grads_nz['params']['actor']['bias']
        # grads['params']['wm']['kernel'] = grads_nz['params']['wm']['kernel']
        # grads['params']['wm']['bias'] = grads_nz['params']['wm']['bias']
        # print('zeroed it')
        # print(jax.tree_map(lambda x: x.shape, grads))
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, metrics

    def sample_test_batch(rng):
        rng, _rng_batch, _rng_aug = split(rng, 3)
        batch = data_utils.sample_batch_from_dataset(_rng_batch, dataset_test, args.bs, args.ctx_len, args.seq_len)
        batch = data_utils.augment_batch(_rng_aug, batch, 0)
        return rng, batch

    def sample_train_batch(rng):
        rng, _rng_batch, _rng_aug = split(rng, 3)
        batch = data_utils.sample_batch_from_dataset(_rng_batch, dataset_train, args.bs, args.ctx_len, args.seq_len)
        batch = data_utils.augment_batch(_rng_aug, batch, args.n_augs, dist=args.aug_dist)
        return rng, batch

    iter_test, iter_train = jax.jit(iter_test), jax.jit(iter_train)
    metrics_before, metrics_train, metrics_test, metrics_after = [], [], [], []
    # --------------------------- BEFORE TRAINING ---------------------------
    pbar = tqdm(range(args.n_iters_eval), desc="Before Training")
    for _ in pbar:
        rng, batch = sample_test_batch(rng)
        train_state, metrics = iter_test(train_state, batch)
        pbar.set_postfix(mse_act=metrics['mse_act'].mean().item(), mse_obs=metrics['mse_obs'].mean().item())
        metrics_before.append(metrics)
    save_pkl(args.save_dir, "metrics_before", tree_stack(metrics_before))

    # --------------------------- TRAINING ---------------------------
    pbar = tqdm(range(args.n_iters + 1), desc="Training")
    for i_iter in pbar:
        if args.save_dir is not None and ((args.n_ckpts > 0 and i_iter == args.n_iters) or (
                args.n_ckpts > 1 and i_iter % (args.n_iters // (args.n_ckpts - 1)) == 0)):
            save_pkl(args.save_dir, f"ckpt_{i_iter:07d}", dict(i_iter=i_iter, params=train_state.params))
            os.system(f"ln -sf \"{args.save_dir}/ckpt_{i_iter:07d}.pkl\" \"{args.save_dir}/ckpt_latest.pkl\"")

        if args.env_id is not None and i_iter % args.n_iters_rollout == 0:
            mse_act, mse_obs, stats = unroll_ed.do_rollout(agent, train_state.params, args.env_id, dataset_train,
                                                           dataset_test, transform_params,
                                                           num_envs=8, video_dir=None, seed=0, ctx_len=args.ctx_len,
                                                           seq_len=args.seq_len)

        rng, batch = sample_train_batch(rng)
        train_state, metrics_train_i = iter_train(train_state, batch)

        if len(metrics_test) > 0:
            train_loss = np.mean([i['loss'] for i in metrics_train[-20:]])
            test_loss = np.mean([i['loss'] for i in metrics_test[-20:]])
            pbar.set_postfix(train_loss=train_loss, test_loss=test_loss)

        if i_iter % 10 == 0:
            metrics_train.append(metrics_train_i)
            rng, batch = sample_test_batch(rng)
            train_state, metrics_test_i = iter_test(train_state, batch)
            metrics_test.append(metrics_test_i)

        if i_iter % 1000 == 0 or i_iter == args.n_iters:
            save_pkl(args.save_dir, "metrics_train", tree_stack(metrics_train))
            save_pkl(args.save_dir, "metrics_test", tree_stack(metrics_test))

    # --------------------------- AFTER TRAINING ---------------------------
    pbar = tqdm(range(args.n_iters_eval), desc="After Training")
    for _ in pbar:
        rng, batch = sample_test_batch(rng)
        train_state, metrics = iter_test(train_state, batch)
        pbar.set_postfix(mse_act=metrics['mse_act'].mean().item(), mse_obs=metrics['mse_obs'].mean().item())
        metrics_after.append(metrics)
    save_pkl(args.save_dir, "metrics_after", tree_stack(metrics_after))


if __name__ == '__main__':
    main(parse_args())
# TODO: keep it mind that multiple dataset makes it much slower. I think its cause of cat operation
