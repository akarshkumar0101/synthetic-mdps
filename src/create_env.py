import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from einops import repeat
from jax.random import split
from tqdm.auto import tqdm

import util
from mdps import smdp, dsmdp, csmdp
from mdps.natural_mdps import DiscretePendulum
from mdps.wrappers import LogWrapper
from mdps.wrappers_mine import TimeLimit, FlattenObservationWrapper


def create_env(env_id):
    env_cfg = dict([sub.split('=') for sub in env_id.split(';')])

    if env_cfg['name'] in gymnax.registered_envs:
        env, env_params = gymnax.make(env_cfg['name'])
        env.sample_params = lambda rng: env_params
    elif env_cfg['name'] == 'DiscretePendulum-v1':
        env = DiscretePendulum()
        env_params = env.default_params
        env.sample_params = lambda rng: env_params
    elif env_cfg['name'] == 'csmdp':
        # d_state, d_obs, n_acts = int(env_cfg['d_state']), int(env_cfg['d_obs']), int(env_cfg['n_acts'])
        i_d = [2, 4, 8, 16, 32][int(env_cfg['i_d'])]
        i_s = [0, 1e-1, 3e-1, 1e0, 3e0][int(env_cfg['i_s'])]
        model_init = csmdp.Init(d_state=i_d, std=i_s)

        t_a = [1, 2, 3, 4, 5][int(env_cfg['t_a'])]
        t_c = [0, 1, 2, 4, 8][int(env_cfg['t_c'])]
        t_l = [1e-2, 3e-2, 1e-1, 3e-1, 1e0][int(env_cfg['t_l'])]
        t_s = [0, 3e-2, 1e-1, 3e-1, 1e0][int(env_cfg['t_s'])]
        model_trans = csmdp.DeltaTransition(d_state=i_d, n_acts=t_a,
                                            n_layers=t_c, d_hidden=16, activation=nn.relu,
                                            locality=t_l, std=t_s)

        o_d = [2, 4, 8, 16, 32][int(env_cfg['o_d'])]
        o_c = [0, 1, 2, 4, 8][int(env_cfg['o_c'])]
        model_obs = csmdp.Observation(d_state=i_d, d_obs=o_d, n_layers=o_c, d_hidden=16, activation=nn.relu, std=0.)

        r_c = [0, 1, 2, 4, 8][int(env_cfg['r_c'])]
        model_rew = csmdp.DenseReward(d_state=i_d, n_layers=r_c, d_hidden=16, activation=nn.relu, std=0.)

        model_done = smdp.NeverDone()
        env = smdp.SyntheticMDP(model_init, model_trans, model_obs, model_rew, model_done)
    elif env_cfg['name'] == 'dsmdp':
        model_init = dsmdp.Init(16, std=1.)
        model_trans = dsmdp.Transition(16, 4)
        model_obs = dsmdp.Observation(16, 8)
        model_rew = dsmdp.DenseReward(16)
        model_done = smdp.NeverDone()

        env = smdp.SyntheticMDP(model_init, model_trans, model_obs, model_rew, model_done)
    else:
        raise NotImplementedError
    env = FlattenObservationWrapper(env)
    # env = ObsNormRand(env)
    if 'tl' in env_cfg:
        env = TimeLimit(env, n_steps_max=int(env_cfg['tl']))
    env = LogWrapper(env)
    return env


def plot_csmdp_env(env_id, env, env_params):
    env_cfg = dict([sub.split('=') for sub in env_id.split(';')])
    assert env_cfg['name'] == 'csmdp'

    n_res = 30
    state_space = env.state_space(env_params)
    assert state_space.shape == (2,)
    n_acts = env.action_space(env_params).n
    cols = ['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w'][:n_acts]

    lim = jnp.stack([jnp.full(state_space.shape, state_space.low), jnp.full(state_space.shape, state_space.high)])
    xlim, ylim = lim.T
    x, y = jnp.linspace(*xlim, n_res), jnp.linspace(*ylim, n_res)
    xm, ym = jnp.meshgrid(x, y, indexing='ij')
    state_all = rearrange(jnp.stack([xm, ym], axis=-1), "i j d -> (i j) d")

    _rng = jax.random.PRNGKey(0)
    rew_all = jax.vmap(env.get_rew, in_axes=(None, 0, None))(_rng, state_all, env_params)
    done_all = jax.vmap(env.is_done, in_axes=(None, 0, None))(_rng, state_all, env_params)
    rew_all = rearrange(rew_all, '(i j) -> i j', i=n_res, j=n_res) # needs to be reversed for pcolormesh
    done_all = rearrange(done_all, '(i j) -> i j', i=n_res, j=n_res).astype(float)
    # plt.pcolormesh(x, y, rew_all, alpha=0.8, vmin=rew_all.min(), vmax=rew_all.max(), cmap='RdYlGn')
    plt.pcolormesh(xm, ym, rew_all, alpha=0.8, vmin=-3., vmax=3., cmap='RdYlGn')

    # akstate = state_all
    # akrew = rew_all.flatten()
    # plt.scatter(*akstate.T, c=akrew, alpha=0.8, vmin=-3., vmax=3., cmap='RdYlGn', s=100)


    plt.colorbar()
    done_all = jnp.stack([jnp.zeros_like(done_all)] * 3 + [done_all], axis=-1)
    plt.pcolormesh(xm, ym, done_all)

    step_fn = jax.vmap(jax.vmap(env.step_env, in_axes=(None, 0, None, None)), in_axes=(None, None, 0, None))
    _rng = jax.random.PRNGKey(0)
    act = jnp.arange(n_acts)
    _, state_all_n, _, _, _ = step_fn(_rng, state_all, act, env_params)

    state, state_n = repeat(state_all, 'ij d -> (a ij) d', a=n_acts), rearrange(state_all_n, 'a ij d -> (a ij) d')
    c = []
    for col in cols:
        c.extend([col] * (n_res * n_res))
    plt.quiver(*state.T, *(state_n - state).T, angles='xy', scale_units='xy', scale=1, width=0.002, color=c)

    obs, state = jax.vmap(env.reset_env, in_axes=(0, None))(split(_rng, 256), env_params)
    plt.scatter(*state.T, color=(0, 0, 0, 0.4), edgecolors='none', label='init state distribution')

    # buffer = random_agent_collect(rng, env, env_params, n_envs=1024, n_steps=256)
    #
    # state = rearrange(buffer['state'], 'N T ... -> (N T) ...')  # (N T) D=2
    # plt.scatter(*state.T, color=(0, 0, 1, 0.01), label='random walk state distribution')
    # state = buffer['state'][:, 0]  # N T D=2
    # plt.scatter(*state.T, color=(0, 0, 0, 1.), label='init state distribution')

    # put legend above figure
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))


def plot_env_id_multi(env_id):
    env_cfg = dict([sub.split('=') for sub in env_id.split(';')])

    if env_cfg['name'] == 'csmdp' and env_cfg['i_d'] == '0':
        plot_env_fn = plot_csmdp_env
    else:
        raise NotImplementedError

    env = create_env(env_id)
    while hasattr(env, '_env'):
        env = env._env

    plt.figure(figsize=(30, 24))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        rng = jax.random.PRNGKey(i)
        env_params = env.sample_params(rng)
        plot_env_fn(env_id, env, env_params)
    plt.suptitle(f"{env_id}", fontsize=20)
    plt.tight_layout()
    return plt.gcf()


def test_unroll():
    env_id = "name=csmdp;i_d=0;i_s=0;t_a=0;t_c=0;t_l=0;t_s=0;o_d=0;o_c=0;r_c=0;tl=32"
    env = create_env(env_id)

    rng = jax.random.PRNGKey(0)
    env_params = env.sample_params(rng)

    # print(jax.tree_map(lambda x: x.shape, env_params))

    reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    step = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))

    n_envs, n_steps = 64, 1024
    rng, _rng = split(rng)
    obs, state = reset(split(_rng, n_envs), env_params)

    for i in tqdm(range(n_steps)):
        rng, _rng = split(rng)
        act = jax.random.randint(_rng, (n_envs,), 0, 3)
        rng, _rng = split(rng)
        obs, state, rew, done, info = step(split(_rng, n_envs), state, act, env_params)


def viz_env(env_id):
    plot_env_id_multi(env_id)

    # while hasattr(env, '_env'):
    #     env = env._env
    # rng = jax.random.PRNGKey(0)
    # env_params = env.sample_params(rng)
    #
    # rng = jax.random.PRNGKey(0)
    # obs, state = env.reset(rng, env_params)
    # obs, state, rew, done, info = env.step(rng, state, 100, env_params)

    # plot_csmdp_env(env_id, env, env_params)
    # plt.show()

    # fig = plot_env_id_multi(env_id)
    # plt.show()
    # fig.savefig(f"../data/viz/main.png")
    # plt.close()

    # for i in tqdm(range(5)):
    # env_id = f"name=csmdp;i_d=0;i_s=2;t_a=2;t_c=2;t_l={i};t_s=0;o_d=0;o_c=0;r_c=2;tl=64"
    # fig = plot_env_id_multi(env_id)
    # fig.savefig(f"../data/viz/{i}.png")
    # plt.close()


def random_agent_collect(rng, env, env_params, n_envs, n_steps):
    reset = jax.vmap(env.reset, in_axes=(0, None))
    step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    rng, _rng = split(rng)
    obs, state = reset(split(_rng, n_envs), env_params)

    buffer = []
    for _ in tqdm(range(n_steps)):
        rng, _rng = split(rng)
        act = jax.random.randint(_rng, (n_envs,), 0, env.action_space(env_params).n)
        rng, _rng = split(rng)
        obs_n, state_n, rew, done, info = step(split(_rng, n_envs), state, act, env_params)
        buffer_i = dict(obs=obs, state=state, act=act, rew=rew, done=done)
        buffer.append(buffer_i)
        obs, state = obs_n, state_n
    buffer = util.tree_stack(buffer)
    buffer = jax.tree_map(lambda x: rearrange(x, 't e ... -> e t ...'), buffer)
    return buffer


from einops import rearrange


def general_plot_env(env, env_params):
    from sklearn.manifold import TSNE
    rng = jax.random.PRNGKey(0)
    buffer = random_agent_collect(rng, env, env_params, 32, 48)
    obs, state, rew, act, done = buffer['obs'], buffer['state'], buffer['rew'], buffer['act'], buffer['done']

    tsne = TSNE(n_components=2)
    vs = tsne.fit_transform(rearrange(state, 'e t d -> (e t) d'))
    vs = rearrange(vs, '(e t) d -> e t d', e=32)

    print(vs.shape)
    print(rew.shape, act.shape, done.shape)

    vs_flat = rearrange(vs, 'e t d -> (e t) d')
    rews_flat = rearrange(rew, 'e t -> (e t)')
    # plt.scatter(*vs_flat.T, c=rews_flat, alpha=0.8, vmin=-3., vmax=3., cmap='RdYlGn')
    # plt.colorbar()

    x = rearrange(state, 'e t d -> (e t) d')
    c = rearrange(rew, 'e t -> (e t)')
    plt.scatter(*x.T, c=c, alpha=0.8, vmin=-3., vmax=3., cmap='RdYlGn')
    plt.colorbar()

    plt.xlim(-3, 3)
    plt.ylim(-3, 3)




if __name__ == "__main__":
    env_id = "name=csmdp;i_d=0;i_s=0;t_a=1;t_c=2;t_l=2;t_s=0;o_d=1;o_c=1;r_c=4;tl=64"
    env = create_env(env_id)
    env = env._env._env

    env_params = env.sample_params(jax.random.PRNGKey(0))
    plt.figure(figsize=(16, 7))
    plt.subplot(1, 2, 1)
    plot_csmdp_env(env_id, env, env_params)
    plt.subplot(1, 2, 2)
    general_plot_env(env, env_params)
    plt.tight_layout()
    plt.show()
