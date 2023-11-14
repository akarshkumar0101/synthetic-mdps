import jax.numpy as jnp
import gymnax


class CartPole(gymnax.environments.CartPole):
    def sample_params(self, rng):
        return self.default_params

    def reset_env(self, rng, params):
        return super().reset_env(rng, params)

    def step_env(self, rng, state, action, params):
        obs, state, rew, done, info = super().step_env(rng, state, action, params)
        # rew = state.theta
        rew = -jnp.abs(state.theta)
        rew = (rew--4.19792)/3.77367
        done = jnp.zeros_like(done)
        return obs, state, rew, done, info


class MountainCar(gymnax.environments.MountainCar):
    def sample_params(self, rng):
        return self.default_params

    def reset_env(self, rng, params):
        return super().reset_env(rng, params)

    def step_env(self, rng, state, action, params):
        obs, state, rew, done, info = super().step_env(rng, state, action, params)
        rew = state.position
        rew = (rew + .52686)/.07063
        done = jnp.zeros_like(done)
        return obs, state, rew, done, info


class Acrobot(gymnax.environments.Acrobot):
    def sample_params(self, rng):
        return self.default_params

    def reset_env(self, rng, params):
        return super().reset_env(rng, params)

    def step_env(self, rng, state, action, params):
        obs, state, rew, done, info = super().step_env(rng, state, action, params)
        rew = -jnp.cos(state.joint_angle1)-jnp.cos(state.joint_angle2 + state.joint_angle1)
        rew = (rew--1.91526)/.11145
        done = jnp.zeros_like(done)
        return obs, state, rew, done, info
