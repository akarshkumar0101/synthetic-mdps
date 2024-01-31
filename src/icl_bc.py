import argparse
import os
import pickle

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from jax import config as jax_config
from jax.random import split
from tqdm.auto import tqdm

import util
from agents.regular_transformer import BCTransformer, WMTransformer

jax_config.update("jax_debug_nans", True)

parser = argparse.ArgumentParser()
parser.add_argument("--entity", type=str, default=None)
parser.add_argument("--project", type=str, default="synthetic-mdps")
parser.add_argument("--name", type=str, default=None)

# parser.add_argument("--env_id", type=str, default="CartPole-v1")
parser.add_argument("--dataset_path", type=str, default=None)
parser.add_argument("--load_dir", type=str, default=None)
parser.add_argument("--save_dir", type=str, default=None)
parser.add_argument("--save_agent", type=lambda x: x == "True", default=False)

parser.add_argument("--n_iters", type=int, default=10000)
parser.add_argument("--n_augs", type=int, default=int(1e9))
parser.add_argument("--curriculum", type=str, default="none")
parser.add_argument("--time_perm", type=lambda x: x == "True", default=False)

parser.add_argument("--bs", type=int, default=256)
# parser.add_argument("--mbs", type=int, default=256)
parser.add_argument("--lr", type=float, default=2.5e-4)
parser.add_argument("--clip_grad_norm", type=float, default=1.)

parser.add_argument("--obj", type=str, default="bc")  # bc or wm


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


def main(args):
    print(args)
    # run = wandb.init(entity=args.entity, project=args.project, name=args.name, config=args)

    with open(args.dataset_path, 'rb') as f:
        dataset = pickle.load(f)  # D T ...
    dataset['obs'] = (dataset['obs'] - dataset['obs'].mean(axis=(0, 1))) / (dataset['obs'].std(axis=(0, 1)) + 1e-5)
    print('Dataset shape: ', jax.tree_map(lambda x: x.shape, dataset))

    ds_size, T, d_obs = dataset['obs'].shape
    n_acts = dataset['act'].max() + 1

    d_obs_uni = 64
    n_acts_uni = 8
    assert n_acts <= n_acts_uni
    n_acts_extra = n_acts_uni - n_acts

    def augment_instance(instance, task_id):
        obs, logits, act = instance['obs'], instance['logits'], instance['act']
        rng = jax.random.PRNGKey(task_id)
        rng, _rng = split(rng)
        obs_mat = jax.random.normal(_rng, (d_obs_uni, d_obs)) * (1 / d_obs)
        rng, _rng = split(rng)
        act_perm = jax.random.permutation(_rng, n_acts_uni)
        i_act_perm = jnp.zeros_like(act_perm)
        i_act_perm = i_act_perm.at[act_perm].set(jnp.arange(n_acts_uni))

        rng, _rng = split(rng)
        if args.time_perm:
            time_perm = jax.random.permutation(_rng, T)
        else:
            time_perm = jnp.arange(T)

        obs_aug = obs @ obs_mat.T
        act_aug = act_perm[act]

        logits_extra = jnp.full((T, n_acts_extra), -jnp.inf)
        logits_aug = jnp.concatenate([logits, logits_extra], axis=-1)
        logits_aug = logits_aug[:, i_act_perm]

        obs_aug, logits_aug, act_aug = obs_aug[time_perm], logits_aug[time_perm], act_aug[time_perm]
        return dict(obs=obs_aug, logits=logits_aug, act=act_aug)

    rng = jax.random.PRNGKey(0)

    if args.obj == 'bc':
        agent = BCTransformer(n_acts=n_acts_uni, n_layers=4, n_heads=4, d_embd=64, n_steps=T)
    elif args.obj == 'wm':
        agent = WMTransformer(n_acts=n_acts_uni, n_layers=4, n_heads=4, d_embd=64, n_steps=T, d_obs=d_obs_uni)
    else:
        raise NotImplementedError

    rng, _rng = split(rng)
    if args.load_dir is not None:
        with open(f"{args.load_dir}/agent_params.pkl", 'rb') as f:
            agent_params = pickle.load(f)
    else:
        batch = {k: dataset[k][0] for k in ['obs', 'logits', 'act']}
        batch = augment_instance(batch, 0)
        agent_params = agent.init(_rng, batch['obs'], batch['act'])

    tx = optax.chain(optax.clip_by_global_norm(args.clip_grad_norm), optax.adam(args.lr, eps=1e-8))
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
        metrics = dict(loss=ce.mean(), ce=ce, kldiv=kldiv, entr=entr, ppl=ppl,
                       tar_entr=tar_entr, tar_ppl=tar_ppl, acc=acc, tar_acc=tar_acc)
        return ce.mean(), metrics

    def loss_fn_wm(agent_params, batch):
        obs_pred = jax.vmap(agent.apply, in_axes=(None, 0, 0))(agent_params, batch['obs'], batch['act'])
        obs = batch['obs']  # B T O
        obs_n = jnp.concatenate([obs[:, 1:], obs[:, :1]], axis=1)  # B T O
        l2 = optax.l2_loss(obs_pred, obs_n).mean(axis=(0, -1))  # T
        metrics = dict(loss=l2.mean(), l2=l2)
        return l2.mean(), metrics

    loss_fn = {'bc': loss_fn_bc, 'wm': loss_fn_wm}[args.obj]

    def do_iter(rng, train_state, n_augs):
        rng, _rng = split(rng)
        task_id = jax.random.randint(_rng, (args.bs,), minval=0, maxval=n_augs)
        rng, _rng = split(rng)
        idx = jax.random.randint(_rng, (args.bs,), minval=0, maxval=ds_size)
        batch = jax.tree_map(lambda x: x[idx], dataset)
        batch = jax.vmap(augment_instance)(batch, task_id)

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params, batch)
        train_state = train_state.apply_gradients(grads=grads)
        return rng, train_state, metrics

    do_iter = jax.jit(do_iter)

    metrics = []
    pbar = tqdm(range(args.n_iters))
    for i_iter in pbar:
        if args.curriculum == 'linear_log':
            n_augs = int(jnp.e ** ((jnp.log(args.n_augs) / args.n_iters) * i_iter))
        else:
            n_augs = args.n_augs

        rng, train_state, metrics_i = do_iter(rng, train_state, n_augs)
        metrics.append(metrics_i)
        pbar.set_postfix(loss=metrics_i['loss'].item())

        # if i_iter % max(1, args.n_iters // 1000) == 0:
        #     wandb_data = dict(n_augs=n_augs,
        #                       tar_acc=metrics_i['tar_acc'].mean(),
        #                       tar_entr=metrics_i['tar_entr'].mean(), tar_ppl=metrics_i['tar_ppl'].mean())
        #     for k in ['ce', 'kldiv', 'entr', 'ppl', 'acc']:
        #         wandb_data[f"{k}"] = metrics_i[k].mean()
        #         wandb_data[f"{k}_first"] = metrics_i[k][0]
        #         wandb_data[f"{k}_last"] = metrics_i[k][-1]
        #     if i_iter % max(1, args.n_iters // 10) == 0:
        #         for k in ['kldiv', 'acc', 'ppl']:
        #             x = jnp.stack([jnp.arange(T), metrics_i[k]], axis=-1)  # (T, 2)
        #             table = wandb.Table(data=x.tolist(), columns=['token_pos', 'val'])
        #             wandb_data[f"{k}_vs_ctx"] = wandb.plot.line(table, 'token_pos', 'val', title=f'{k} vs token_pos')
        #     wandb.log(wandb_data, step=i_iter)
    metrics = util.tree_stack(metrics)

    if args.save_dir is not None:
        os.makedirs(f"{args.save_dir}/", exist_ok=True)
        with open(f"{args.save_dir}/metrics.pkl", 'wb') as f:
            pickle.dump(metrics, f)
        if args.save_agent:
            with open(f"{args.save_dir}/agent_params.pkl", 'wb') as f:
                pickle.dump(train_state.params, f)

    # wandb.finish()


if __name__ == '__main__':
    main(parse_args())
