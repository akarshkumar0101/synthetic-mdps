import jax.numpy as jnp
import gymnax
from gridenv import GridEnv


class CartPole(gymnax.environments.CartPole):
    def __init__(self):
        super().__init__()
        self.n_acts = self.num_actions

    def sample_params(self, rng):
        return self.default_params

    def reset_env(self, rng, params):
        return super().reset_env(rng, params)

    def step_env(self, rng, state, action, params):
        obs, state, rew, done, info = super().step_env(rng, state, action, params)
        # rew = state.theta
        rew = -jnp.abs(state.theta)
        # rew = (rew--4.19792)/3.77367
        done = jnp.zeros_like(done)
        info["base_env_state"] = state
        return obs, state, rew, done, info


class MountainCar(gymnax.environments.MountainCar):
    def __init__(self):
        super().__init__()
        self.n_acts = self.num_actions

    def sample_params(self, rng):
        return self.default_params

    def reset_env(self, rng, params):
        return super().reset_env(rng, params)

    def step_env(self, rng, state, action, params):
        obs, state, rew, done, info = super().step_env(rng, state, action, params)
        rew = state.position
        # rew = (rew--.52686)/.07063
        done = jnp.zeros_like(done)
        info["base_env_state"] = state
        return obs, state, rew, done, info


class Acrobot(gymnax.environments.Acrobot):
    def __init__(self):
        super().__init__()
        self.n_acts = self.num_actions

    def sample_params(self, rng):
        return self.default_params

    def reset_env(self, rng, params):
        return super().reset_env(rng, params)

    def step_env(self, rng, state, action, params):
        obs, state, rew, done, info = super().step_env(rng, state, action, params)
        rew = -jnp.cos(state.joint_angle1) - jnp.cos(state.joint_angle2 + state.joint_angle1)
        # rew = (rew--1.91526)/.11145
        done = jnp.zeros_like(done)
        info["base_env_state"] = state
        return obs, state, rew, done, info


def wtf(env, rng, n_envs, n_steps):
    rng, _rng = split(rng)
    env_params = jax.vmap(env.sample_params)(split(_rng, n_envs))

    rng, _rng = split(rng)
    obs, state = jax.vmap(env.reset)(split(_rng, n_envs), env_params)

    rews = []
    for t in range(n_steps):
        rng, _rng = split(rng)
        act = jax.random.randint(_rng, (n_envs,), 0, env.n_acts)
        rng, _rng = split(rng)
        obs, state, rew, done, info = jax.vmap(env.step)(split(_rng, n_envs), state, act, env_params)
        rews.append(rew)
    rews = jnp.stack(rews, axis=1)
    return rews

if __name__ == "__main__":
    import jax
    from jax.random import split
    import matplotlib.pyplot as plt

    rng = jax.random.PRNGKey(0)
    env = GridEnv(grid_len=8)

    rews = 


    # rng = jax.random.PRNGKey(0)
    # rng, _rng = split(rng)
    # a = jax.random.randint(_rng, (100000, 2), 0, 8)
    # rng, _rng = split(rng)
    # b = jax.random.randint(_rng, (100000, 2), 0, 8)
    # d = (a-b)
    # print(jnp.linalg.norm(d, axis=-1).mean())
    # print(jnp.sqrt((d**2).sum(axis=-1)).mean())



    plt.plot(rews.mean(axis=0))
    plt.show()
