import pickle
from functools import partial

import gymnax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from einops import rearrange
from flax.training.train_state import TrainState
from jax.random import split
from tqdm.auto import tqdm

from agents.basic import BasicAgentSeparate
from agents.regular_transformer import Transformer
from agents.util import DenseObsEmbed
from algos.ppo_dr import PPO
from mdps.wrappers import LogWrapper
from util import tree_stack


# class MountainCarDenseRew(MyGymnaxWrapper):
#
#     def reset_env(self, key, params):
#         obs, state = self._env.reset_env(key, params)
#         return obs, state
#
#     def step_env(self, key, state, action, params):
#         obs, state, rew, done, info = self._env.step_env(key, state, action, params)
#         # pos_range = jnp.array([-1.2, 0.6])
#         # vel_range = jnp.array([-0.07, 0.07])
#         r = jnp.array([0.6 - -1.2, 0.07 - -0.07])
#         mid = jnp.array([-jnp.pi / 6, 0.])
#         a = jnp.array([state.position, state.velocity])
#         a = ((a - mid) / r)
#         a = jnp.linalg.norm(a)
#         rew = a
#         return obs, state, rew, done, info


def main():
    rng = jax.random.PRNGKey(0)
    env, env_params = gymnax.make('CartPole-v1')
    env.sample_params = lambda rng: env_params
    env = LogWrapper(env)
    n_acts = env.action_space(env_params).n

    ObsEmbed = partial(DenseObsEmbed, d_embd=128)
    agent = BasicAgentSeparate(ObsEmbed, n_acts)

    ppo = PPO(agent, env, sample_env_params=env.sample_params,
              n_envs=4, n_steps=128, n_updates=16, n_envs_batch=1, lr=2.5e-4, clip_grad_norm=0.5,
              clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, gamma=0.99, gae_lambda=0.95)
    init_agent_env = jax.jit(jax.vmap(ppo.init_agent_env))
    ppo_step = jax.jit(jax.vmap(ppo.ppo_step, in_axes=(0, None)))
    eval_step = jax.jit(jax.vmap(ppo.eval_step, in_axes=(0, None)))

    rng, _rng = split(rng)
    carry = init_agent_env(split(_rng, 1))

    rets = []
    for i_iter in tqdm(range(100)):
        carry, buffer = eval_step(carry, None)
        rets.append(buffer['info']['returned_episode_returns'].mean(axis=(-1, -2)))

    for i_iter in tqdm(range(900)):
        carry, buffer = ppo_step(carry, None)
        rets.append(buffer['info']['returned_episode_returns'].mean(axis=(-1, -2)))

    eval_stuff = []
    for i_iter in tqdm(range(256 * 64)):
        carry, buffer = eval_step(carry, None)
        rets.append(buffer['info']['returned_episode_returns'].mean(axis=(-1, -2)))

        ks = ['env_state', 'obs', 'logits', 'act', 'rew', 'done']
        eval_stuff.append({k: buffer[k] for k in ks})
    rets = rearrange(jnp.stack(rets), 'N S -> S N')
    # print(jax.tree_map(lambda x: x.shape, buffer))

    eval_stuff = tree_stack(eval_stuff)
    print(jax.tree_map(lambda x: x.shape, eval_stuff))
    eval_stuff = jax.tree_map(lambda x: rearrange(x, 'N 1 T E ... -> (N E) T ...'), eval_stuff)  # only 0th seed
    print(jax.tree_map(lambda x: x.shape, eval_stuff))

    with open('../data/temp/cartpole_data.pkl', 'wb') as f:
        pickle.dump(eval_stuff, f)

    plt.plot(rets.T, c=[0.1, 0.1, 0.1, 0.1])
    plt.plot(rets.mean(axis=0), label='mean')
    plt.legend()
    plt.ylabel('Return')
    plt.xlabel('Training Iteration')

    plt.show()


# def kl_div(logits, logits_target, axis=-1):
#     log_p, log_q = jax.nn.log_softmax(logits_target), jax.nn.log_softmax(logits)
#     return (jnp.exp(log_p) * (log_p - log_q)).sum(axis=axis)


def main2():
    with open('../data/temp/cartpole_data.pkl', 'rb') as f:
        dataset = pickle.load(f)
    obs = dataset['obs']
    obs = (obs - obs.mean(axis=(0, 1), keepdims=True)) / obs.std(axis=(0, 1), keepdims=True)
    dataset['obs'] = obs

    print(jax.tree_map(lambda x: x.shape, dataset))

    ds_size, T, O = dataset['obs'].shape
    B = 256
    A = dataset['act'].max() + 1
    n_tasks = 1000000

    O_universal = 4
    A_universal = 2
    assert A <= A_universal
    A_extra = A_universal - A

    def augment_instance(instance, task_id):
        obs, logits, act = instance['obs'], instance['logits'], instance['act']

        rng = jax.random.PRNGKey(task_id)

        rng, _rng = split(rng)
        obs_mat = jax.random.normal(_rng, (O_universal, O)) * (1 / O)

        rng, _rng = split(rng)
        act_perm = jax.random.permutation(_rng, A_universal)

        rng, _rng = split(rng)
        time_perm = jax.random.permutation(_rng, T)
        # time_perm = jnp.arange(T)

        obs_aug = obs @ obs_mat.T
        act_aug = act_perm[act]

        logits_extra = jnp.full((T, A_extra), -jnp.inf)
        logits = jnp.concatenate([logits, logits_extra], axis=-1)
        logits = logits[:, act_perm]
        # logits = jax.nn.log_softmax(logits, axis=-1)

        return dict(obs=obs_aug[time_perm], logits=logits[time_perm], act=act_aug[time_perm])

    rng = jax.random.PRNGKey(0)

    agent = Transformer(n_acts=A_universal, n_layers=4, n_heads=4, d_embd=64, n_steps=T)

    rng, _rng = split(rng)

    batch = {k: dataset[k][0] for k in ['obs', 'logits', 'act']}
    batch = augment_instance(batch, 0)
    agent_params = agent.init(_rng, batch['obs'], batch['act'])

    tx = optax.chain(optax.clip_by_global_norm(1.), optax.adam(3e-4, eps=1e-8))
    train_state = TrainState.create(apply_fn=agent.apply, params=agent_params, tx=tx)

    def do_iter(rng, train_state):
        rng, _rng = split(rng)
        task_id = jax.random.randint(_rng, (B,), minval=0, maxval=n_tasks)
        rng, _rng = split(rng)
        idx = jax.random.randint(_rng, (B,), minval=0, maxval=ds_size)
        batch = {k: dataset[k][idx] for k in ['obs', 'logits', 'act']}
        batch = jax.vmap(augment_instance)(batch, task_id)

        def loss_fn(agent_params, batch):
            logits, val = jax.vmap(agent.apply, in_axes=(None, 0, 0))(agent_params, batch['obs'], batch['act'])

            # loss = jax.scipy.special.rel_entr(jax.nn.softmax(batch['logits']), jax.nn.softmax(logits)).sum(axis=-1)

            ce_label = optax.softmax_cross_entropy_with_integer_labels(logits, batch['act'])
            kl_div = optax.kl_divergence(jax.nn.log_softmax(logits), jax.nn.softmax(batch['logits']))
            ce_dist = optax.softmax_cross_entropy(jax.nn.log_softmax(logits), jax.nn.softmax(batch['logits']))
            tar_entr = optax.softmax_cross_entropy(jax.nn.log_softmax(batch['logits']), jax.nn.softmax(batch['logits']))

            loss = ce_dist
            return loss.mean(), dict(kl_div=kl_div, ce_label=ce_label, ce_dist=ce_dist, tar_entr=tar_entr)

        (_, data), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params, batch)
        train_state = train_state.apply_gradients(grads=grads)
        return rng, train_state, data

    do_iter = jax.jit(do_iter)

    pbar = tqdm(range(1000))
    for i_iter in pbar:
        rng, train_state, data = do_iter(rng, train_state)
        # pbar.set_postfix(loss=loss.mean(), loss0=loss[:, 0].mean(), loss1=loss[:, -1].mean())
        # pbar.set_description(f'ppl={jnp.e ** loss.mean(): 6.4f} tar_ppl={jnp.e ** tar_entr.mean(): 6.4f}')
        kl_div, ce_label, ce_dist, tar_entr = data['kl_div'], data['ce_label'], data['ce_dist'], data['tar_entr']
        pbar.set_description(f'kl0={kl_div[:, 0].mean(): 6.4f} kl1={kl_div[:, -1].mean(): 6.4f}')

    plt.plot(kl_div.mean(axis=0))
    plt.ylabel('Loss')
    plt.xlabel('In context timesteps')
    plt.title('Randomized env timesteps')
    # plt.axhline(jnp.e ** tar_entr.mean(), c='r')
    plt.show()


if __name__ == '__main__':
    main2()
