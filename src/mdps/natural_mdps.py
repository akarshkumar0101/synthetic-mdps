import gymnax
import jax.numpy as jnp
from gymnax.environments.classic_control import Pendulum
from gymnax.environments.spaces import Discrete


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


class DiscretePendulum(Pendulum):
    def step_env(self, key, state, action, params):
        action = jnp.array([-2., -1., 0., 1., 2.])[action]
        return super().step_env(key, state, action, params)

    @property
    def num_actions(self) -> int:
        return 5

    def action_space(self, params):
        return Discrete(5)


if __name__ == "__main__":
    import jax
    from src.mdps.wrappers_mine import collect_random_agent, GaussianObsReward

    rng = jax.random.PRNGKey(0)
    # env = GridEnv(grid_len=8)
    env = Acrobot()

    rews = collect_random_agent(env, rng, 1024, 128)
    print(rews.mean(), rews.std())

    env = GaussianObsReward(env, 1024, 128)

    rews = collect_random_agent(env, rng, 1024, 128)
    print(rews.mean(), rews.std())
