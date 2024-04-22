import argparse
import os
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange, repeat
from jax.random import split

import run
from mdps.smdp import SyntheticMDP


def plot_env_dynamics(env, env_params, xlim=(-3, 3), ylim=(-3, 3)):
    n_acts = env.action_space(env_params).n

    x, y = jnp.linspace(*xlim, 30), jnp.linspace(*ylim, 30)
    x, y = jnp.meshgrid(x, y, indexing='ij')
    state = rearrange(jnp.stack([x, y], axis=-1), "i j s -> (i j) s")
    act = jnp.arange(n_acts)

    _rng = jax.random.PRNGKey(0)
    step_fn = partial(env.step, params=env_params)
    step_fn = jax.vmap(jax.vmap(step_fn, in_axes=(None, 0, None)), in_axes=(None, None, 0))
    _, state_n, _, _, _ = step_fn(_rng, state, act)
    state = repeat(state, 'ij s -> a ij s', a=n_acts)
    state, state_n = rearrange(state, 'a ij s -> (a ij) s'), rearrange(state_n, 'a ij s -> (a ij) s')
    c = ['r'] * 900 + ['g'] * 900 + ['b'] * 900
    plt.quiver(*state.T, *(state_n - state).T, angles='xy', scale_units='xy', scale=1, width=0.002, color=c)


def plot_start_end_states(env, env_params):
    model_init, model_rew = env.model_init, env.model_rew
    state_start = env_params['params_init']['params']['state_start']
    dist_thresh = model_init.dist_thresh
    circle_start = plt.Circle(state_start, dist_thresh, color=[0, 0, 1, .3], fill=True)
    plt.gca().add_patch(circle_start)

    state_goal = env_params['params_rew']['params']['state_goal']
    dist_thresh = model_rew.dist_thresh
    circle_goal = plt.Circle(state_goal, dist_thresh, color=[0, 1, 0, .3], fill=True)
    plt.gca().add_patch(circle_goal)


def plot_trajectory(states):
    plt.scatter(*states.T, c=np.arange(len(states)), s=70, cmap='brg')
    plt.colorbar()


# def plot_trajectory(states):
#     from einops import rearrange
#     from matplotlib.collections import LineCollection
#     from matplotlib.colors import BoundaryNorm, ListedColormap
#
#     points = rearrange(states, "t d -> t 1 d")
#     dydx = np.linspace(0, 1, len(states))
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)
#
#     norm = plt.Normalize(dydx.min(), dydx.max())
#     lc = LineCollection(segments, cmap='brg', norm=norm)
#     lc.set_array(dydx)
#     lc.set_linewidth(10)
#     line = plt.gca().add_collection(lc)
#     plt.gcf().colorbar(line, ax=plt.gca())


# ------------------------------ DON'T NEED ANYTHING ABOVE THIS LINE ------------------------------


def random_agent_collect(rng, env, env_params, n_envs=1024, n_steps=256):
    n_acts = env.action_space(env_params).n

    def step(carry, _):
        rng, state = carry
        rng, _rng = split(rng)
        act = jax.random.randint(_rng, (n_envs,), 0, n_acts)
        rng, _rng = split(rng)
        obs, state_n, rew, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(split(_rng, n_envs), state, act,
                                                                                    env_params)
        carry = rng, state_n
        return carry, dict(obs=obs, state=state, act=act, rew=rew, done=done, info=info)

    rng, _rng = split(rng)
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(split(_rng, n_envs), env_params)
    carry = rng, state
    carry, buffer = jax.lax.scan(step, carry, jnp.arange(n_steps))
    buffer = jax.tree_map(lambda x: rearrange(x, 'T N ... -> N T ...'), buffer)
    return buffer


def plot_env(rng, env, env_params):
    state_space = env.state_space(env_params)
    assert state_space.shape == (2,)
    n_acts = env.action_space(env_params).n
    cols = ['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w'][:n_acts]

    lim = jnp.stack([jnp.ones(state_space.shape) * state_space.low, jnp.ones(state_space.shape) * state_space.high])
    xlim, ylim = lim.T

    nlen = 30
    x, y = jnp.linspace(*xlim, nlen), jnp.linspace(*ylim, nlen)
    x, y = jnp.meshgrid(x, y, indexing='ij')
    state_all = rearrange(jnp.stack([x, y], axis=-1), "i j s -> (i j) s")
    act = jnp.arange(n_acts)

    step_fn = partial(env.step, params=env_params)
    step_fn = jax.vmap(jax.vmap(step_fn, in_axes=(None, 0, None)), in_axes=(None, None, 0))
    rng, _rng = split(rng)
    _, state_all_n, _, _, _ = step_fn(_rng, state_all, act)

    state, state_n = repeat(state_all, 'ij d -> (a ij) d', a=n_acts), rearrange(state_all_n, 'a ij d -> (a ij) d')
    c = []
    for col in cols:
        c.extend([col] * (nlen * nlen))
    plt.quiver(*state.T, *(state_n - state).T, angles='xy', scale_units='xy', scale=1, width=0.002, color=c)

    rew_all = jax.vmap(env.get_rew, in_axes=(0, None))(state_all, env_params)
    done_all = jax.vmap(env.is_done, in_axes=(0, None))(state_all, env_params)
    plt.scatter(*state_all.T, c=rew_all, cmap='RdYlGn', alpha=0.3, vmin=rew_all.min(), vmax=rew_all.max(), s=200.,
                label='reward', marker='s', edgecolor='none')
    plt.colorbar()

    # done_all = done_all.at[:100].set(True)
    c = jnp.stack([jnp.zeros_like(done_all), jnp.zeros_like(done_all), jnp.zeros_like(done_all), done_all / 2.],
                  axis=-1)
    plt.scatter(*state_all.T, c=c, s=200., label='done', marker='x')  # , edgecolor='none')

    rng, _rng = split(rng)
    buffer = random_agent_collect(_rng, env, env_params, n_envs=128, n_steps=256)

    state = rearrange(buffer['state'], 'N T ... -> (N T) ...')  # (N T) D=2
    plt.scatter(*state.T, color=(0, 0, 1, 0.01), label='random walk state distribution')
    state = buffer['state'][:, 0]  # N T D=2
    plt.scatter(*state.T, color=(0, 0, 0, 1.), label='init state distribution')

    plt.legend()


parser = argparse.ArgumentParser()
# experiment args
parser.add_argument("--env_id", type=str, default=None)
# name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=linear;mrl=4x64
parser.add_argument("--save_dir", type=str, default=None)


def main(args):
    env = run.create_env(args.env_id)
    while not isinstance(env, SyntheticMDP):
        env = env._env
    assert isinstance(env, SyntheticMDP)

    n_plts = 3
    plt.figure(figsize=(6 * n_plts, 5 * n_plts))
    for i_plt in range(n_plts * n_plts):
        rng = jax.random.PRNGKey(i_plt)
        env_params = env.sample_params(rng)
        plt.subplot(n_plts, n_plts, i_plt + 1)
        plot_env(rng, env, env_params)

    plt.suptitle(f"env_id: {args.env_id}")

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        plt.savefig(f"{args.save_dir}/env_viz.png")


if __name__ == '__main__':
    main(parser.parse_args())
