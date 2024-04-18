import argparse
import glob
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from jax import config as jax_config
from jax.random import split
from tqdm.auto import tqdm

import util
from agents.regular_transformer import BCTransformer, WMTransformer

jax_config.update("jax_debug_nans", True)

parser = argparse.ArgumentParser()
# parser.add_argument("--entity", type=str, default=None)
# parser.add_argument("--project", type=str, default="synthetic-mdps")
# parser.add_argument("--name", type=str, default=None)

# parser.add_argument("--env_id", type=str, default="CartPole-v1")
parser.add_argument("--dataset_paths", type=str, nargs="+", default=[])
parser.add_argument("--exclude_dataset_paths", type=str, nargs="+", default=[])
parser.add_argument("--percent_dataset", type=float, default=1.0)

parser.add_argument("--load_dir", type=str, default=None)
parser.add_argument("--save_dir", type=str, default=None)
parser.add_argument("--save_agent", type=lambda x: x == "True", default=False)
parser.add_argument("--n_ckpts", type=int, default=0)

parser.add_argument("--n_iters_eval", type=int, default=500)
parser.add_argument("--n_iters", type=int, default=10000)

parser.add_argument("--n_augs", type=int, default=0)
parser.add_argument("--p_same", type=float, default=0)
parser.add_argument("--time_perm", type=lambda x: x == "True", default=False)

parser.add_argument("--bs", type=int, default=64)
# parser.add_argument("--mbs", type=int, default=256)
parser.add_argument("--lr", type=float, default=2.5e-4)
parser.add_argument("--clip_grad_norm", type=float, default=1.)

parser.add_argument("--obj", type=str, default="bc")  # bc or wm

# Model arguments
group = parser.add_argument_group("model")
group.add_argument("--n_layers", type=int, default=4)
group.add_argument("--n_heads", type=int, default=4)
group.add_argument("--d_embd", type=int, default=256)
group.add_argument("--ctx_len", type=int, default=256)
group.add_argument("--mask_type", type=str, default="causal")
group.add_argument("--d_obs_uni", type=int, default=64)
group.add_argument("--n_acts_uni", type=int, default=10)


# parser.add_argument("--curriculum", type=str, default="none")
# n_augs = int(jnp.e ** ((jnp.log(args.n_augs) / args.n_iters) * i_iter))

def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    # set all "none" to None
    for k, v in vars(args).items():
        if v == "None":
            setattr(args, k, None)
    return args


def calc_entropy_stable(logits, axis=-1):
    logits = jax.nn.log_softmax(logits, axis=axis)
    probs = jnp.exp(logits)
    logits = jnp.where(probs == 0, 0., logits)  # replace -inf with 0
    return -(probs * logits).sum(axis=axis)


def preprocess_dataset(dataset, d_obs_uni, n_acts_uni):
    assert 'obs' in dataset and 'logits' in dataset and 'act' in dataset
    assert dataset['obs'].ndim == 3 and dataset['logits'].ndim == 3 and dataset['act'].ndim == 2
    dataset = jax.tree_map(lambda x: jnp.asarray(x), dataset)  # convert to jax

    dataset['obs'] = (dataset['obs'] - dataset['obs'].mean(axis=(0, 1))) / (dataset['obs'].std(axis=(0, 1)) + 1e-5)
    ds_size, T, d_obs = dataset['obs'].shape
    n_acts = dataset['act'].max() + 1
    assert n_acts <= n_acts_uni
    n_acts_extra = n_acts_uni - n_acts

    rng = jax.random.PRNGKey(0)
    obs_mat = jax.random.orthogonal(rng, n=max(d_obs, d_obs_uni), shape=())[:d_obs_uni, :d_obs]
    dataset['obs'] = dataset['obs'] @ obs_mat.T
    logits_extra = jnp.full((ds_size, T, n_acts_extra), -jnp.inf)
    dataset['logits'] = jnp.concatenate([dataset['logits'], logits_extra], axis=-1)

    dataset = jax.tree_map(lambda x: np.asarray(x), dataset)  # convert back to numpy # TODO: manage devices
    return dataset


def augment_batch(rng, batch, n_augs, p_same=0.0, do_time_perm=False):
    if n_augs == 0:
        return batch

    bs, T, d_obs = batch['obs'].shape
    _, _, n_acts = batch['logits'].shape

    def augment_instance(instance, aug_id):
        rng = jax.random.PRNGKey(aug_id)
        _rng_obs, _rng_act, _rng_time = split(rng, 3)
        obs_mat = jax.random.normal(_rng_obs, (d_obs, d_obs)) * jnp.sqrt(1. / d_obs)
        act_perm = jax.random.permutation(_rng_act, n_acts)
        i_act_perm = jnp.zeros_like(act_perm)
        i_act_perm = i_act_perm.at[act_perm].set(jnp.arange(n_acts))
        time_perm = jax.random.permutation(_rng_time, T) if do_time_perm else jnp.arange(T)
        obs = (instance['obs'] @ obs_mat.T)[time_perm]
        logits = (instance['logits'][:, i_act_perm])[time_perm]
        act = (act_perm[instance['act']])[time_perm]
        return dict(obs=obs, logits=logits, act=act)

    rng, _rng = split(rng)
    aug_ids = jax.random.randint(_rng, (bs,), minval=0, maxval=n_augs)
    rng, _rng = split(rng)
    same_aug_mask = jax.random.uniform(_rng, (bs,)) < p_same
    aug_ids = jnp.where(same_aug_mask, 0, aug_ids)
    return jax.vmap(augment_instance)(batch, aug_ids)


def sample_batch_from_dataset(rng, dataset, bs):
    rng, _rng = split(rng)
    idx = jax.random.randint(_rng, (bs,), minval=0, maxval=len(dataset['obs']))
    batch = jax.tree_map(lambda x: x[idx], dataset)
    return batch


def sample_batch_from_datasets(rng, datasets, bs):
    rng, _rng = split(rng)
    i_ds = jax.random.randint(_rng, (bs,), minval=0, maxval=len(datasets))

    batches = []
    for i, ds in enumerate(datasets):
        bs_ds = jnp.sum(i_ds == i).item()
        rng, _rng = split(rng)
        i = jax.random.randint(_rng, (bs_ds,), minval=0, maxval=len(ds['obs']))
        batch = jax.tree_map(lambda x: x[i], ds)
        batches.append(batch)
    batch = util.tree_cat(batches)
    return batch


def save_pkl(save_dir, name, item):
    if save_dir is not None:
        os.makedirs(f"{save_dir}/", exist_ok=True)
        with open(f"{save_dir}/{name}.pkl", 'wb') as f:
            pickle.dump(item, f)


def main(args):
    print(args)
    if args.n_ckpts > 0:
        assert args.n_iters % args.n_ckpts == 0
    # run = wandb.init(entity=args.entity, project=args.project, name=args.name, config=args)

    include_paths = [os.path.abspath(p) for i in args.dataset_paths for p in glob.glob(i)]
    exclude_paths = [os.path.abspath(p) for i in args.exclude_dataset_paths for p in glob.glob(i)]
    dataset_paths = sorted(set(include_paths) - set(exclude_paths))

    print(f"Found {len(dataset_paths)} datasets")

    datasets = []
    for p in dataset_paths:
        print(f"Loading dataset from {p}")
        with open(p, 'rb') as f:
            dataset = pickle.load(f)
        dataset = preprocess_dataset(dataset, d_obs_uni=args.d_obs_uni, n_acts_uni=args.n_acts_uni)
        print(f"Dataset i shape: {jax.tree_map(lambda x: x.shape, dataset)}")
        datasets.append(dataset)
    # dataset = util.tree_cat(datasets)
    dataset = {k: np.concatenate([d[k] for d in datasets], axis=0) for k in datasets[0].keys()}
    rng = jax.random.PRNGKey(0)
    idx_ds = jax.random.permutation(rng, len(dataset['obs']))
    idx_ds = idx_ds[:int(len(idx_ds) * args.percent_dataset)]
    dataset_small = jax.tree_map(lambda x: x[idx_ds], dataset)

    print("----------------------------")
    print(f"Full Dataset shape: {jax.tree_map(lambda x: x.shape, dataset)}")
    print(f"Small Dataset shape: {jax.tree_map(lambda x: x.shape, dataset_small)}")

    rng = jax.random.PRNGKey(0)
    if args.obj == 'bc':
        agent = BCTransformer(n_acts=args.n_acts_uni, n_layers=args.n_layers, n_heads=args.n_heads,
                              d_embd=args.d_embd, n_steps=args.ctx_len, mask_type=args.mask_type)
    elif args.obj == 'wm':
        agent = WMTransformer(n_acts=args.n_acts_uni, n_layers=args.n_layers, n_heads=args.n_heads,
                              d_embd=args.d_embd, n_steps=args.ctx_len, d_obs=args.d_obs_uni)
    else:
        raise NotImplementedError

    rng, _rng = split(rng)
    if args.load_dir is not None:
        with open(f"{args.load_dir}/ckpt_final.pkl.pkl", 'rb') as f:
            ckpt = pickle.load(f)
            agent_params = ckpt['params']
    else:
        # batch = sample_batch_from_datasets(rng, datasets, 1)
        batch = sample_batch_from_dataset(rng, dataset, 1)
        batch = augment_batch(rng, batch, n_augs=1, do_time_perm=False)
        batch = jax.tree_map(lambda x: x[0], batch)
        agent_params = agent.init(_rng, batch['obs'], batch['act'])

    lr_warmup = optax.linear_schedule(0., args.lr, args.n_iters // 100)
    lr_main = optax.constant_schedule(args.lr)
    lr_schedule = optax.join_schedules([lr_warmup, lr_main], [args.n_iters // 100])
    tx = optax.chain(optax.clip_by_global_norm(args.clip_grad_norm), optax.adam(lr_schedule, eps=1e-8))
    # tx = optax.chain(optax.clip_by_global_norm(args.clip_grad_norm), optax.adam(args.lr, eps=1e-8))
    train_state = TrainState.create(apply_fn=agent.apply, params=agent_params, tx=tx)

    def loss_fn_bc(agent_params, batch):
        logits = jax.vmap(agent.apply, in_axes=(None, 0, 0))(agent_params, batch['obs'], batch['act'])
        # ce_label = optax.softmax_cross_entropy_with_integer_labels(logits, batch['act'])
        ce = optax.softmax_cross_entropy(jax.nn.log_softmax(logits), jax.nn.softmax(batch['logits'])).mean(axis=0)
        kldiv = optax.kl_divergence(jax.nn.log_softmax(logits), jax.nn.softmax(batch['logits'])).mean(axis=0)
        entr = calc_entropy_stable(logits).mean(axis=0)
        ppl = jnp.exp(entr)
        tar_entr = calc_entropy_stable(batch['logits']).mean(axis=0)
        tar_ppl = jnp.exp(tar_entr)
        acc = jnp.mean(batch['act'] == logits.argmax(axis=-1), axis=0)
        tar_acc = jnp.mean(batch['act'] == batch['logits'].argmax(axis=-1), axis=0)
        loss = ce.mean()
        metrics = dict(loss=loss, ce=ce, kldiv=kldiv, entr=entr, ppl=ppl,
                       tar_entr=tar_entr, tar_ppl=tar_ppl, acc=acc, tar_acc=tar_acc)
        return loss, metrics

    def loss_fn_wm(agent_params, batch):
        obs_pred = jax.vmap(agent.apply, in_axes=(None, 0, 0))(agent_params, batch['obs'], batch['act'])
        obs = batch['obs']  # B T O
        obs_n = jnp.concatenate([obs[:, 1:], obs[:, :1]], axis=1)  # B T O
        l2 = optax.l2_loss(obs_pred, obs_n).mean(axis=(0, -1))  # T
        loss = l2.mean()
        metrics = dict(loss=loss, l2=l2)
        return loss, metrics

    loss_fn = {'bc': loss_fn_bc, 'wm': loss_fn_wm}[args.obj]

    def iter_eval(train_state, batch):
        loss, metrics = loss_fn(train_state.params, batch)
        return train_state, metrics

    def iter_step(train_state, batch):
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params, batch)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, metrics

    iter_eval, iter_step = jax.jit(iter_eval), jax.jit(iter_step)
    metrics_before, metrics_train, metrics_after = [], [], []
    metrics_train_testset = []

    # --------------------------- BEFORE TRAINING ---------------------------
    pbar = tqdm(range(args.n_iters_eval), desc="Before")
    for _ in pbar:
        rng, _rng_batch, _rng_aug = split(rng, 3)
        batch = sample_batch_from_dataset(_rng_batch, dataset, args.bs)
        batch = augment_batch(_rng_aug, batch, n_augs=args.n_augs, p_same=0., do_time_perm=args.time_perm)

        train_state, metrics = iter_eval(train_state, batch)
        pbar.set_postfix(loss=metrics['loss'].item())
        metrics_before.append(metrics)
    metrics_before = util.tree_stack(metrics_before)
    save_pkl(args.save_dir, "metrics_before.pkl", metrics_before)

    # --------------------------- TRAINING ---------------------------
    pbar = tqdm(range(args.n_iters), desc="Training")
    for i_iter in pbar:
        if args.n_ckpts > 0 and i_iter % (args.n_iters // args.n_ckpts) == 0:
            i_ckpt = i_iter // (args.n_iters // args.n_ckpts)
            save_pkl(args.save_dir, f"ckpt_{i_ckpt}.pkl", dict(i_ckpt=i_ckpt, i_iter=i_iter, params=train_state.params))

        rng, _rng_batch, _rng_aug = split(rng, 3)
        batch = sample_batch_from_dataset(_rng_batch, dataset_small, args.bs)
        batch = augment_batch(_rng_aug, batch, n_augs=args.n_augs, p_same=args.p_same, do_time_perm=args.time_perm)
        train_state, metrics = iter_step(train_state, batch)
        pbar.set_postfix(loss=metrics['loss'].item(), n_augs=args.n_augs)
        metrics_train.append(metrics)

        if i_iter % 10 == 0:
            batch = sample_batch_from_dataset(_rng_batch, dataset, args.bs)
            batch = augment_batch(_rng_aug, batch, n_augs=args.n_augs, p_same=0., do_time_perm=args.time_perm)
            train_state, metrics = iter_eval(train_state, batch)
            metrics_train_testset.append(metrics)

    metrics_train = util.tree_stack(metrics_train)
    save_pkl(args.save_dir, "metrics_train.pkl", metrics_train)

    metrics_train_testset = util.tree_stack(metrics_train_testset)
    save_pkl(args.save_dir, "metrics_train_testset.pkl", metrics_train_testset)

    # --------------------------- AFTER TRAINING ---------------------------
    pbar = tqdm(range(args.n_iters_eval), desc="After")
    for _ in pbar:
        rng, _rng_batch, _rng_aug = split(rng, 3)
        batch = sample_batch_from_dataset(_rng_batch, dataset, args.bs)
        batch = augment_batch(_rng_aug, batch, n_augs=args.n_augs, p_same=0., do_time_perm=args.time_perm)

        train_state, metrics = iter_eval(train_state, batch)
        pbar.set_postfix(loss=metrics['loss'].item())
        metrics_after.append(metrics)
    metrics_after = util.tree_stack(metrics_after)
    save_pkl(args.save_dir, "metrics_after.pkl", metrics_after)
    if args.save_agent:
        save_pkl(args.save_dir, "ckpt_final.pkl",
                 dict(i_ckpt=args.n_ckpts, i_iter=args.n_iters, params=train_state.params))
    save_pkl(args.save_dir, "hi", 'hi')


if __name__ == '__main__':
    main(parse_args())
# TODO: keep it mind that multiple dataset makes it much slower. I think its cause of cat operation
