import argparse
import glob
import os
# print('CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'])
# print('XLA_PYTHON_CLIENT_PREALLOCATE', os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'])
# print('XLA_PYTHON_CLIENT_MEM_FRACTION', os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'])
import pickle

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from einops import repeat
from flax.training.train_state import TrainState
from jax import config as jax_config
from jax.random import split
from tqdm.auto import tqdm

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
group.add_argument("--percent_dataset", type=float, default=1.0)
# group.add_argument("--n_augs_test", type=int, default=0)
group.add_argument("--n_augs", type=int, default=0)
group.add_argument("--aug_dist", type=str, default="uniform")
# group.add_argument("--time_perm", type=lambda x: x == "True", default=False)
# group.add_argument("--zipf", type=lambda x: x == "True", default=False)

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


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if v == "None":
            setattr(args, k, None)  # set all "none" to None
    if args.mini_bs is None:
        args.mini_bs = args.bs
    assert args.bs % args.mini_bs == 0
    return args


def load_dataset(path):
    with open(path, "rb") as f:
        dataset = pickle.load(f)
    assert 'obs' in dataset and ('logits' in dataset) or ('act_mean' in dataset)
    assert dataset['obs'].ndim == 3
    dataset = jax.tree_map(lambda x: np.array(x).astype(np.float32), dataset)
    act_key = 'logits' if 'logits' in dataset else 'act_mean'
    obs, act, rew, done = dataset['obs'], dataset[act_key], dataset['rew'], dataset['done']
    N, T, Do = obs.shape
    time = repeat(jnp.arange(T), 'T -> N T', N=N)
    return dict(obs=obs, act=act, rew=rew, done=done, time=time)


def train_test_split(dataset, percent_train=0.8):
    N = len(dataset['obs'])
    n_train = int(N * percent_train)
    dataset_train = jax.tree_map(lambda x: x[:n_train], dataset)
    dataset_test = jax.tree_map(lambda x: x[n_train:], dataset)
    return dataset_train, dataset_test


def get_percent_dataset(dataset, percent_dataset_vert=0.1, percent_dataset_horz=0.1):
    N, T, _ = dataset['obs'].shape
    n, t = int(N * percent_dataset_vert), int(T * percent_dataset_horz)
    return jax.tree_map(lambda x: x[:n, :t], dataset)


def transform_dataset(dataset, obs_mean, obs_std, act_mean, act_std, d_obs_uni, d_act_uni):
    d_obs, d_act = dataset['obs'].shape[-1], dataset['act'].shape[-1]
    obs, act, rew, done, time = dataset['obs'], dataset['act'], dataset['rew'], dataset['done'], dataset['time']
    obs = (obs - obs_mean) / (obs_std + 1e-8)
    act = (act - act_mean) / (act_std + 1e-8)

    rng = jax.random.PRNGKey(0)
    obs_mat = jax.random.orthogonal(rng, max(d_obs, d_obs_uni))[:d_obs_uni, :d_obs]  # not square probably
    act_mat = jax.random.orthogonal(rng, max(d_act, d_act_uni))[:d_act_uni, :d_act]
    obs_mat = obs_mat / np.sqrt(np.diag(obs_mat @ obs_mat.T))[:, None]  # to make sure output is standard normal
    act_mat = act_mat / np.sqrt(np.diag(act_mat @ act_mat.T))[:, None]
    obs_mat, act_mat = np.array(obs_mat, dtype=np.float32), np.array(act_mat, dtype=np.float32)
    dataset = dict(obs=obs @ obs_mat.T, act=act @ act_mat.T, rew=rew, done=done, time=time)
    return dataset


def construct_dataset(include_paths, exclude_paths, d_obs_uni, d_acts_uni, percent_dataset=(1., 1.)):
    include_paths = [os.path.abspath(p) for i in include_paths for p in glob.glob(i)]
    exclude_paths = [os.path.abspath(p) for i in exclude_paths for p in glob.glob(i)]
    paths = sorted(set(include_paths) - set(exclude_paths))
    print(f"Found {len(paths)} datasets")
    assert len(paths) > 0
    datasets_train, datasets_test = [], []
    for path in paths:
        print(f"Loading dataset from {path}")
        dataset = load_dataset(path)
        print(f"Dataset shape: {jax.tree_map(lambda x: (x.shape, x.dtype), dataset)}")
        dataset_train, dataset_test = train_test_split(dataset, percent_train=0.8)
        obs_mean, obs_std = dataset_train['obs'].mean(axis=(0, 1)), dataset_train['obs'].std(axis=(0, 1))
        act_mean, act_std = dataset_train['act'].mean(axis=(0, 1)), dataset_train['act'].std(axis=(0, 1))

        pv, ph = percent_dataset
        dataset_train = get_percent_dataset(dataset_train, percent_dataset_vert=pv, percent_dataset_horz=ph)

        dataset_train = transform_dataset(dataset_train, obs_mean, obs_std, act_mean, act_std, d_obs_uni, d_acts_uni)
        dataset_test = transform_dataset(dataset_test, obs_mean, obs_std, act_mean, act_std, d_obs_uni, d_acts_uni)
        datasets_train.append(dataset_train)
        datasets_test.append(dataset_test)
    dataset_train = {k: np.concatenate([d[k] for d in datasets_train], axis=0) for k in datasets_train[0].keys()}
    dataset_test = {k: np.concatenate([d[k] for d in datasets_test], axis=0) for k in datasets_test[0].keys()}
    return dataset_train, dataset_test, (obs_mean, obs_std, act_mean, act_std)


def sample_batch_from_dataset(rng, dataset, batch_size, ctx_len, seq_len):
    rng, _rng1, _rng2 = split(rng, 3)
    n_e, n_t, *_ = dataset['obs'].shape
    i_e = jax.random.randint(_rng1, (batch_size,), minval=0, maxval=n_e)
    i_t = jax.random.randint(_rng2, (batch_size,), minval=0, maxval=n_t - seq_len)

    def get_instance_i_h(rng):
        return jax.random.permutation(rng, seq_len)[:ctx_len].sort()

    i_h = jax.vmap(get_instance_i_h)(split(rng, batch_size))

    i_e = i_e[:, None]
    i_t = i_t[:, None] + i_h
    batch = jax.tree_map(lambda x: x[i_e, i_t, ...], dataset)
    return batch


# def sample_batch_from_dataset(rng, dataset, batch_size, ctx_len):
#     rng, _rng1, _rng2 = split(rng, 3)
#     n_e, n_t, *_ = dataset['obs'].shape
#     i_e = jax.random.randint(_rng1, (batch_size,), minval=0, maxval=n_e)
#     i_t = jax.random.randint(_rng2, (batch_size,), minval=0, maxval=n_t - ctx_len)
#     batch = jax.tree_map(lambda x: x[i_e[:, None], (i_t[:, None] + jnp.arange(ctx_len)), ...], dataset)
#     return batch

# def sample_batch_from_datasets(rng, datasets, bs):
#     rng, _rng = split(rng)
#     i_ds = jax.random.randint(_rng, (bs,), minval=0, maxval=len(datasets))
#
#     batches = []
#     for i, ds in enumerate(datasets):
#         bs_ds = jnp.sum(i_ds == i).item()
#         rng, _rng = split(rng)
#         i = jax.random.randint(_rng, (bs_ds,), minval=0, maxval=len(ds['obs']))
#         batch = jax.tree_map(lambda x: x[i], ds)
#         batches.append(batch)
#     batch = tree_cat(batches)
#     return batch


# def subsample_dataset(rng, dataset, percent_dataset=1.0):
#     idx_ds = jax.random.permutation(rng, len(dataset['obs']))
#     idx_ds = idx_ds[:int(len(idx_ds) * percent_dataset)]
#     dataset_small = jax.tree_map(lambda x: x[idx_ds], dataset)


# def augment_batch(rng, batch, n_augs, p_same=0.0, do_time_perm=False, zipf=False):
#     if n_augs == 0:
#         return batch
#     bs, T, d_obs = batch['obs'].shape
#     _, _, n_acts = batch['logits'].shape
#
#     def augment_instance(instance, aug_id):
#         rng = jax.random.PRNGKey(aug_id)
#         _rng_obs, _rng_act, _rng_time = split(rng, 3)
#         obs_mat = jax.random.normal(_rng_obs, (d_obs, d_obs)) * jnp.sqrt(1. / d_obs)
#         act_perm = jax.random.permutation(_rng_act, n_acts)
#         i_act_perm = jnp.zeros_like(act_perm)
#         i_act_perm = i_act_perm.at[act_perm].set(jnp.arange(n_acts))
#         time_perm = jax.random.permutation(_rng_time, T) if do_time_perm else jnp.arange(T)
#         obs = (instance['obs'] @ obs_mat.T)[time_perm]
#         logits = (instance['logits'][:, i_act_perm])[time_perm]
#         act = (act_perm[instance['act']])[time_perm]
#         return dict(obs=obs, logits=logits, act=act)
#
#     rng, _rng = split(rng)
#     if zipf:
#         p = 1 / jnp.arange(1, n_augs + 1)
#         p = p / p.sum()
#         aug_ids = jax.random.choice(_rng, jnp.arange(n_augs), (bs,), p=p)
#     else:
#         aug_ids = jax.random.randint(_rng, (bs,), minval=0, maxval=n_augs)
#         rng, _rng = split(rng)
#         same_aug_mask = jax.random.uniform(_rng, (bs,)) < p_same
#         aug_ids = jnp.where(same_aug_mask, 0, aug_ids)
#     return jax.vmap(augment_instance)(batch, aug_ids)


# def augment_batch(rng, batch,
#                   n_augs_obs, p_same_obs,
#                   n_augs_act, p_same_act,
#                   n_augs_time, p_same_time):
#     if n_augs_obs == 0 and n_augs_act == 0 and n_augs_time == 0:
#         return batch
#     bs, T, d_obs = batch['obs'].shape
#     _, _, n_acts = batch['logits'].shape
#
#     def augment_instance(instance, aug_id_obs, aug_id_act, aug_id_time):
#         if n_augs_obs > 0:
#             obs_mat = jax.random.normal(jax.random.PRNGKey(aug_id_obs), (d_obs, d_obs)) * jnp.sqrt(1. / d_obs)
#         else:
#             obs_mat = jnp.eye(d_obs)
#         if n_augs_act > 0:
#             act_perm = jax.random.permutation(jax.random.PRNGKey(aug_id_act), n_acts)
#         else:
#             act_perm = jnp.arange(n_acts)
#         i_act_perm = jnp.zeros_like(act_perm)
#         i_act_perm = i_act_perm.at[act_perm].set(jnp.arange(n_acts))
#         time_perm = jax.random.permutation(jax.random.PRNGKey(aug_id_time), T) if n_augs_time > 0 else jnp.arange(T)
#         obs = (instance['obs'] @ obs_mat.T)[time_perm]
#         logits = (instance['logits'][:, i_act_perm])[time_perm]
#         act = (act_perm[instance['act']])[time_perm]
#         return dict(obs=obs, logits=logits, act=act)
#
#     def get_aug_ids(rng, n_augs, p_same):
#         rng1, rng2 = split(rng)
#         if int(p_same) == -1:
#             p = 1 / jnp.arange(1, n_augs + 1)
#             p = p / p.sum()
#             aug_ids = jax.random.choice(rng1, jnp.arange(n_augs), (bs,), p=p)
#         else:
#             aug_ids = jax.random.randint(rng1, (bs,), minval=0, maxval=n_augs)
#             if isinstance(p_same, float):
#                 same_aug_mask = jax.random.uniform(rng2, (bs,)) < p_same
#                 aug_ids = jnp.where(same_aug_mask, 0, aug_ids)
#         return aug_ids
#
#     rng, _rng1, _rng2 = split(rng, 3)
#     aug_ids_obs = get_aug_ids(_rng1, n_augs_obs, p_same_obs)
#     aug_ids_act = get_aug_ids(_rng2, n_augs_act, p_same_act)
#     aug_ids_time = get_aug_ids(rng, n_augs_time, p_same_time)
#     return jax.vmap(augment_instance)(batch, aug_ids_obs, aug_ids_act, aug_ids_time)


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
        return dict(obs=instance['obs'] @ obs_mat.T, act=instance['act'] @ act_mat.T,
                    rew=instance['rew'], done=instance['done'], time=instance['time'])

    if dist == "uniform":
        aug_ids = jax.random.randint(rng, (bs,), minval=0, maxval=n_augs)
    elif dist == "zipf":
        p = 1 / jnp.arange(1, n_augs + 1)
        aug_ids = jax.random.choice(rng, jnp.arange(n_augs), (bs,), p=p / p.sum())
    else:
        raise NotImplementedError

    return jax.vmap(augment_instance)(batch, aug_ids)


def main(args):
    print(args)
    args.n_augs_obs = args.n_augs
    args.n_augs_act = args.n_augs
    rng = jax.random.PRNGKey(args.seed)
    # run = wandb.init(entity=args.entity, project=args.project, name=args.name, config=args)

    dataset_train, dataset_test, _ = construct_dataset(args.dataset_paths, args.exclude_dataset_paths,
                                                       args.d_obs_uni, args.d_act_uni,
                                                       percent_dataset=(args.percent_dataset, 1.))
    print("----------------------------")
    print(f"Train Dataset shape: {jax.tree_map(lambda x: (type(x), x.shape, x.dtype), dataset_train)}")
    print(f"Test Dataset shape: {jax.tree_map(lambda x: (type(x), x.shape, x.dtype), dataset_test)}")

    agent = BCTransformer(d_obs=args.d_obs_uni, d_act=args.d_act_uni,
                          n_layers=args.n_layers, n_heads=args.n_heads, d_embd=args.d_embd, ctx_len=args.ctx_len,
                          mask_type=args.mask_type)

    rng, _rng = split(rng)
    if args.load_ckpt is not None:
        with open(args.load_ckpt, "rb") as f:
            agent_params = pickle.load(f)['params']
    else:
        batch = sample_batch_from_dataset(rng, dataset_train, 1, args.ctx_len, args.seq_len)
        batch = augment_batch(rng, batch, 0)
        batch = jax.tree_map(lambda x: x[0], batch)
        agent_params = agent.init(_rng, batch['obs'], batch['act'], batch['time'])
    print("Agent parameter count: ", sum(p.size for p in jax.tree_util.tree_leaves(agent_params)))
    tabulate_fn = nn.tabulate(agent, jax.random.key(0), compute_flops=True, compute_vjp_flops=True)
    batch = sample_batch_from_dataset(rng, dataset_train, args.bs, args.ctx_len, args.seq_len)
    batch = jax.tree_map(lambda x: x[0], (batch['obs'], batch['act'], batch['time']))
    print(tabulate_fn(*batch))

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
        metrics = dict(loss=loss, mse_act=mse_act, mse_obs=mse_obs)
        return loss, metrics

    def iter_test(train_state, batch):
        loss, metrics = loss_fn(train_state.params, batch)
        return train_state, metrics

    def iter_train(train_state, batch):
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params, batch)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, metrics

    def sample_test_batch(rng):
        rng, _rng_batch, _rng_aug = split(rng, 3)
        batch = sample_batch_from_dataset(_rng_batch, dataset_test, args.bs, args.ctx_len, args.seq_len)
        batch = augment_batch(_rng_aug, batch, 0)
        return rng, batch

    def sample_train_batch(rng):
        rng, _rng_batch, _rng_aug = split(rng, 3)
        batch = sample_batch_from_dataset(_rng_batch, dataset_train, args.bs, args.ctx_len, args.seq_len)
        batch = augment_batch(_rng_aug, batch, args.n_augs, dist=args.aug_dist)
        return rng, batch

    iter_test, iter_train = jax.jit(iter_test), jax.jit(iter_train)
    metrics_before, metrics_train, metrics_test, metrics_after = [], [], [], []
    # --------------------------- BEFORE TRAINING ---------------------------
    pbar = tqdm(range(args.n_iters_eval), desc="Before Training")
    for _ in pbar:
        rng, batch = sample_test_batch(rng)
        train_state, metrics = iter_test(train_state, batch)
        pbar.set_postfix(loss=metrics['loss'].item())
        metrics_before.append(metrics)
    save_pkl(args.save_dir, "metrics_before", tree_stack(metrics_before))

    # --------------------------- TRAINING ---------------------------
    pbar = tqdm(range(args.n_iters), desc="Training")
    for i_iter in pbar:
        if (args.n_ckpts - 1) > 0 and i_iter % (args.n_iters // (args.n_ckpts - 1)) == 0:
            save_pkl(args.save_dir, f"ckpt_{i_iter:07d}", dict(i_iter=i_iter, params=train_state.params))
            # if os.path.exists(f"{args.save_dir}/ckpt_latest.pkl"):
            #     os.remove(f"{args.save_dir}/ckpt_latest.pkl")
            # os.symlink(f"{args.save_dir}/ckpt_{i_iter:07d}.pkl", f"{args.save_dir}/ckpt_latest.pkl")

        rng, batch = sample_train_batch(rng)
        train_state, metrics = iter_train(train_state, batch)

        if len(metrics_test) > 0:
            train_loss = np.mean([i['loss'] for i in metrics_train[-20:]])
            test_loss = np.mean([i['loss'] for i in metrics_test[-20:]])
            pbar.set_postfix(train_loss=train_loss, test_loss=test_loss)

        if i_iter % 10 == 0:
            metrics_train.append(metrics)
            rng, batch = sample_test_batch(rng)
            train_state, metrics = iter_test(train_state, batch)
            metrics_test.append(metrics)
        if i_iter % 1000 == 0 or i_iter == args.n_iters - 1:
            save_pkl(args.save_dir, "metrics_train", tree_stack(metrics_train))
            save_pkl(args.save_dir, "metrics_test", tree_stack(metrics_test))

    if args.n_ckpts > 0 and args.save_dir is not None:
        save_pkl(args.save_dir, f"ckpt_{args.n_iters:07d}", dict(i_iter=args.n_iters, params=train_state.params))
        # if os.path.exists(f"{args.save_dir}/ckpt_latest.pkl"):
        #     os.remove(f"{args.save_dir}/ckpt_latest.pkl")
        # os.symlink(f"{args.save_dir}/ckpt_{args.n_iters:07d}.pkl", f"{args.save_dir}/ckpt_latest.pkl")

    # --------------------------- AFTER TRAINING ---------------------------
    pbar = tqdm(range(args.n_iters_eval), desc="After Training")
    for _ in pbar:
        rng, batch = sample_test_batch(rng)
        train_state, metrics = iter_test(train_state, batch)
        pbar.set_postfix(loss=metrics['loss'].item())
        metrics_after.append(metrics)
    save_pkl(args.save_dir, "metrics_after", tree_stack(metrics_after))


if __name__ == '__main__':
    main(parse_args())
# TODO: keep it mind that multiple dataset makes it much slower. I think its cause of cat operation
