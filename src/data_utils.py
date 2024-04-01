import glob
import os
# print('CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'])
# print('XLA_PYTHON_CLIENT_PREALLOCATE', os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'])
# print('XLA_PYTHON_CLIENT_MEM_FRACTION', os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'])
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from einops import repeat
from jax.random import split


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


def get_dataset_transform_params(rng, dataset, d_obs_uni, d_act_uni):
    d_obs, d_act = dataset['obs'].shape[-1], dataset['act'].shape[-1]
    obs_mean, obs_std = dataset['obs'].mean(axis=(0, 1)), dataset['obs'].std(axis=(0, 1))
    act_mean, act_std = dataset['act'].mean(axis=(0, 1)), dataset['act'].std(axis=(0, 1))

    rng, _rng1, _rng2 = split(rng, 3)
    obs_mat = jax.random.orthogonal(_rng1, max(d_obs, d_obs_uni))[:d_obs_uni, :d_obs]  # not square probably
    act_mat = jax.random.orthogonal(_rng2, max(d_act, d_act_uni))[:d_act_uni, :d_act]
    obs_scale = np.sqrt(np.diag(obs_mat @ obs_mat.T))
    act_scale = np.sqrt(np.diag(act_mat @ act_mat.T))
    obs_mat_forward = obs_mat / obs_scale[:, None]  # to make sure output is standard normal
    act_mat_forward = act_mat / act_scale[:, None]
    obs_mat_inverse = obs_mat.T * obs_scale[None, :]
    act_mat_inverse = act_mat.T * act_scale[None, :]
    return dict(obs_mean=obs_mean, obs_std=obs_std, act_mean=act_mean, act_std=act_std,
                obs_mat_forward=obs_mat_forward, act_mat_forward=act_mat_forward,
                obs_mat_inverse=obs_mat_inverse, act_mat_inverse=act_mat_inverse)


def get_identity_transform_params(dataset):
    d_obs, d_act = dataset['obs'].shape[-1], dataset['act'].shape[-1]
    obs_mean, obs_std = jnp.zeros(d_obs), jnp.ones(d_obs)
    act_mean, act_std = jnp.zeros(d_act), jnp.ones(d_act)
    obs_mat = jnp.eye(d_obs)
    act_mat = jnp.eye(d_act)
    return dict(obs_mean=obs_mean, obs_std=obs_std, act_mean=act_mean, act_std=act_std,
                obs_mat_forward=obs_mat, act_mat_forward=act_mat,
                obs_mat_inverse=obs_mat, act_mat_inverse=act_mat)


def transform_obs(obs, transform_params):
    obs_mean, obs_std = transform_params['obs_mean'], transform_params['obs_std']
    obs_mat = transform_params['obs_mat_forward']

    obs = (obs - obs_mean) / (obs_std + 1e-8)
    obs = obs @ obs_mat.T
    return obs


def transform_act(act, transform_params):
    act_mean, act_std = transform_params['act_mean'], transform_params['act_std']
    act_mat = transform_params['act_mat_forward']

    act = (act - act_mean) / (act_std + 1e-8)
    act = act @ act_mat.T
    return act


def transform_dataset(dataset, transform_params):
    dataset_new = dataset.copy()
    dataset_new['obs'] = transform_obs(dataset['obs'], transform_params)
    dataset_new['act'] = transform_act(dataset['act'], transform_params)
    return dataset_new


def inverse_transform_obs(obs, transform_params):
    obs_mean, obs_std = transform_params['obs_mean'], transform_params['obs_std']
    obs_mat = transform_params['obs_mat_inverse']

    obs = obs @ obs_mat.T
    obs = obs * obs_std + obs_mean
    return obs


def inverse_transform_act(act, transform_params):
    act_mean, act_std = transform_params['act_mean'], transform_params['act_std']
    act_mat = transform_params['act_mat_inverse']

    act = act @ act_mat.T
    act = act * act_std + act_mean
    return act


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
