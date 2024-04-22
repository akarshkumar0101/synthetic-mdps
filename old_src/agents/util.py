import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange


class Agent(nn.Module):
    def __call__(self, state, x):  # state.shape: (...), x.shape: (T, ...)
        raise NotImplementedError

    def forward_parallel(self, state, x):  # state.shape: (...), x.shape: (T, ...)
        return self(state, x)

    def forward_recurrent(self, state, x):  # state.shape: (...), x.shape: (...)
        x = jax.tree_map(lambda x: rearrange(x, '... -> 1 ...'), x)
        state, x = self(state, x)
        x = jax.tree_map(lambda x: rearrange(x, '1 ... -> ...'), x)
        return state, x

    def init_state(self, rng):
        return None


class DenseObsEmbed(Agent):
    d_embd: int

    @nn.compact
    def __call__(self, state, x):  # state.shape: (...), x.shape: (T, ...)
        x = rearrange(x, 'T ... -> T (...)')
        x = nn.Dense(features=self.d_embd)(x)  # D
        return state, x


class MinAtarObsEmbed(Agent):
    d_embd: int

    @nn.compact
    def __call__(self, state, x):  # state.shape: (...), x.shape: (T, ...)
        x = nn.Conv(features=8, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=8, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = rearrange(x, 'T ... -> T (...)')
        x = nn.Dense(features=self.d_embd)(x)  # D
        return state, x


class ObsActRewTimeEmbed(Agent):
    d_embd: int
    ObsEmbed: nn.Module
    n_acts: int
    n_steps_max: int

    def setup(self):
        self.obs_embed = self.ObsEmbed()
        self.act_embed = nn.Embed(self.n_acts, features=self.d_embd)
        self.rew_embed = nn.Dense(features=self.d_embd)
        self.time_embed = nn.Embed(self.n_steps_max, features=self.d_embd)

    def __call__(self, state, x):  # state.shape: (...), x.shape: (T, ...)
        oe_state, time = state['oe_state'], state['time']
        obs, act_p, rew_p, done = x['obs'], x['act_p'], x['rew_p'], x['done']

        T, = act_p.shape
        time = time + jnp.arange(T)
        time = time - jax.lax.associative_scan(jnp.maximum, time * done)

        oe_state, x_obs = self.obs_embed(oe_state, obs)  # T, D
        x_act, x_rew, x_time = self.act_embed(act_p), self.rew_embed(rew_p[..., None]), self.time_embed(time)  # T, D
        x = x_obs + x_act + x_rew + x_time
        time = time[-1] + 1
        state = dict(oe_state=oe_state, time=time)
        return state, (x, done)

    def init_state(self, rng):
        oe_state = self.obs_embed.init_state(rng)
        time = jnp.zeros((), dtype=jnp.int32)
        return dict(oe_state=oe_state, time=time)


def main():
    import gymnax
    from gymnax.environments.spaces import Discrete
    for env_id in gymnax.registered_envs:
        env, env_params = gymnax.make(env_id, )
        act_space = env.action_space(env_params)
        obs_space = env.observation_space(env_params)
        if isinstance(act_space, Discrete):
            print(env_id)
            print(f"{obs_space.shape} => {act_space.n}")
            print()



if __name__ == "__main__":
    from jax.random import split
    from functools import partial

    rng = jax.random.PRNGKey(0)
    x = jnp.zeros((5, 10, 10, 6))

    net = ObsActRewTimeEmbed(d_embd=64, ObsEmbed=MinAtarObsEmbed, n_acts=4, n_steps_max=100)

    rng, _rng = split(rng)

    method = partial(net.init_with_output, method=net.init_state)
    state, params = jax.vmap(method, in_axes=(None, 0))(rng, split(_rng, 4))
    print(jax.tree_map(lambda x: x.shape, state))
    print(jax.tree_map(lambda x: x.shape, params))

    # state, params = net.apply({}, rng, rngs={"params": rng}, mutable=True, method=net.init_state)
    # print(state)
    # print(jax.tree_map(lambda x: x.shape, params))

    # state, y = net.apply(params, None, x)
    # print(jax.tree_map(lambda x: x.shape, params))
    # print(jax.tree_map(lambda x: x.shape, y))
    # print(f"param_count: {sum(x.size for x in jax.tree_util.tree_leaves(params))}")
