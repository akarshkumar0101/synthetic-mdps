import flax.linen as nn
import jax
import jax.numpy as jnp


class DenseObsEmbed(nn.Module):
    d_embd: int

    @nn.compact
    def __call__(self, obs):  # x: (...)
        x = obs.flatten()
        x = nn.Dense(features=self.d_embd)(x)  # D
        return x


class MinAtarObsEmbed(nn.Module):
    d_embd: int

    @nn.compact
    def __call__(self, obs):  # x: (...)
        x = obs
        x = nn.Conv(features=8, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=8, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.flatten()
        x = nn.Dense(features=self.d_embd)(x)  # D
        return x


class ObsActRewTimeEmbed(nn.Module):
    d_embd: int
    n_acts: int
    n_steps_max: int
    raw_obs_embed: nn.Module

    @nn.compact
    def __call__(self, state, x):
        # state: (), x: (T, ...)
        obs, act_p, rew_p, done = x['obs'], x['act_p'], x['rew_p'], x['done']
        T, = act_p.shape
        time = state + jnp.arange(T)
        time = time - jax.lax.associative_scan(jnp.maximum, time * done)

        x_obs = nn.Dense(features=self.d_embd)(obs)  # T, D
        x_act = nn.Embed(self.n_acts, features=self.d_embd)(act_p)  # T, D
        x_rew = nn.Dense(features=self.d_embd)(rew_p[..., None])  # T, D
        x_time = nn.Embed(self.n_steps_max, features=self.d_embd)(time)  # T, D
        x = x_obs + x_act + x_rew + x_time
        state = time[-1] + 1
        return state, (x, done)

    # def initialize_carry(self, rng):
    #     return jnp.zeros((), dtype=jnp.int32)


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


env_id2obs_embed = {
    "CartPole-v1": DenseObsEmbed,
    "Acrobot-v1": DenseObsEmbed,
    "MountainCar-v0": DenseObsEmbed,
    "Asterix-MinAtar": MinAtarObsEmbed,
    "Breakout-MinAtar": MinAtarObsEmbed,
    "Freeway-MinAtar": MinAtarObsEmbed,
    "SpaceInvaders-MinAtar": MinAtarObsEmbed,
}

if __name__ == "__main__":
    # rng = jax.random.PRNGKey(0)
    # x = jnp.zeros((10, 10, 6))
    #
    # net = MinAtarObsEmbed(64)
    #
    # y, params = net.init_with_output(rng, x)
    # print(jax.tree_map(lambda x: x.shape, params))
    # print(jax.tree_map(lambda x: x.shape, y))
    # param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    # print(f"param_count: {param_count}")
    main()

