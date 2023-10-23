import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces


class GridEnv(environment.Environment):
    def __init__(self, grid_len, max_steps=100):
        super().__init__()
        self.grid_len, self.max_steps = grid_len, max_steps
        self.obs = jnp.zeros((grid_len, grid_len), dtype=jnp.float32) - 1.
        self.action_map = jnp.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        self.n_acts = len(self.action_map)

    def sample_params(self, rng):
        pos_rew = jax.random.randint(rng, (2,), 0, self.grid_len)
        params = dict(pos_rew=pos_rew)
        return params

    # @property
    # def default_params(self) -> EnvParams:
    #     return EnvParams()

    def step_env(self, rng, state, action, params):
        """Performs step transitions in the environment."""
        pos, time = state['pos'], state['time']
        pos = jnp.clip(pos + self.action_map[action], 0, self.grid_len - 1)
        state = dict(pos=pos, time=time + 1)
        obs = self.get_obs(state)
        rew = self.get_rew(state, params)

        done = state['time'] >= self.max_steps
        info = {}
        return obs, state, rew, done, info

    def reset_env(self, rng, params):
        """Performs resetting of environment."""
        pos = jnp.zeros((2,), dtype=jnp.int32)
        state = dict(pos=pos, time=jnp.array(0, dtype=jnp.int32))
        obs = self.get_obs(state)
        return obs, state

    def get_obs(self, state):
        y, x = state['pos']
        obs = self.obs.at[y, x].set(1.)
        return obs

    def get_rew(self, state, params):
        pos = state['pos']
        pos_rew = params['pos_rew']
        d = jnp.linalg.norm((pos - pos_rew).astype(jnp.float32))
        return 1 / (d ** 2 + 1)

    @property
    def name(self) -> str:
        return "GridEnv"

    @property
    def num_actions(self) -> int:
        return 4

    def action_space(self, params):
        return spaces.Discrete(self.n_acts)

    def observation_space(self, params):
        return spaces.Box(-1, 1, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params):
        return spaces.Box(-1, 1, self.state_shape, dtype=jnp.float32)


def main():
    import matplotlib.pyplot as plt
    rng = jax.random.PRNGKey(0)

    env = GridEnv(4, 5)

    rng, _rng = jax.random.split(rng)
    env_params = env.sample_params(_rng)

    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng, env_params)

    print(obs, state)

    for i in range(3):
        print()
        print()
        for t in range(5):
            # print(state['time'])
            rng, _rng = jax.random.split(rng)
            act = jax.random.randint(_rng, (), 0, env.n_acts)
            rng, _rng = jax.random.split(rng)
            obs, state, rew, done, info = env.step(_rng, state, act, env_params)
            # print(state['time'])
            print(done)




if __name__ == '__main__':
    main()
