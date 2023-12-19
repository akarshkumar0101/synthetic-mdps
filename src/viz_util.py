import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from einops import rearrange, repeat


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
    # state, state_n = rearrange(state, 'a ij s -> (a ij) s'), rearrange(state_n, 'a ij s -> (a ij) s')
    c = ['r'] * 900 + ['g'] * 900 + ['b'] * 900
    x, y, xn, yn = state[..., 0].flatten(), state[..., 1].flatten(), state_n[..., 0].flatten(), state_n[..., 1].flatten()
    plt.quiver(x, y, xn - x, yn - y, angles='xy', scale_units='xy', scale=1, width=0.002, color=c)


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
    plt.scatter(*states.T, c=np.arange(len(states)), s=1000, cmap='brg')
    plt.colorbar()
