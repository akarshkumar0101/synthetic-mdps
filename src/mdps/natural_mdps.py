import jax.numpy as jnp
import gymnax


# TODO: rename to DenseCartPole
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


if __name__ == "__main__":
    import jax
    import numpy as np
    from jax.random import split
    import matplotlib.pyplot as plt
    from src.mdps.wrappers_mine import collect_random_agent, GaussianObsReward

    rng = jax.random.PRNGKey(0)
    # env = GridEnv(grid_len=8)
    env = Acrobot()

    rews = collect_random_agent(env, rng, 1024, 128)
    print(rews.mean(), rews.std())

    env = GaussianObsReward(env, 1024, 128)

    rews = collect_random_agent(env, rng, 1024, 128)
    print(rews.mean(), rews.std())
