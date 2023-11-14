import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces


class GridEnv(environment.Environment):
    def __init__(self, grid_len, start_state='random'):
        super().__init__()
        self.grid_len = grid_len
        self.obs_shape = (grid_len, grid_len)
        self.start_state = start_state
        self.obs = jnp.zeros((grid_len, grid_len), dtype=jnp.float32)
        self.action_map = jnp.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        self.n_acts = len(self.action_map)

    def sample_params(self, rng):
        if self.start_state == 'random':
            rng, _rng = jax.random.split(rng)
            pos_start = jax.random.randint(_rng, (2,), 0, self.grid_len)
        else:
            pos_start = jnp.zeros((2,), dtype=jnp.int32)
        rng, _rng = jax.random.split(rng)
        pos_rew = jax.random.randint(_rng, (2,), 0, self.grid_len)
        params = dict(pos_start=pos_start, pos_rew=pos_rew)
        return params

    # @property
    # def default_params(self) -> EnvParams:
    #     return EnvParams()

    def reset_env(self, rng, params):
        """Performs resetting of environment."""
        state = params['pos_start']
        obs = self.get_obs(state)
        return obs, state

    def step_env(self, rng, state, action, params):
        """Performs step transitions in the environment."""
        state = jnp.clip(state + self.action_map[action], 0, self.grid_len - 1)
        obs = self.get_obs(state)
        rew = self.get_rew(state, params)
        rew = (rew - 0.12123) / 0.16537

        done = jnp.zeros((), dtype=jnp.bool_)
        info = {}
        return obs, state, rew, done, info

    def get_obs(self, state):
        y, x = state
        obs = self.obs.at[y, x].set(1.)
        return obs

    def get_rew(self, state, params):
        a, b = state, params['pos_rew']
        d = jnp.linalg.norm(a-b, axis=-1)
        return 1 / (d ** 2 + 1)
    # 1/(d**2+1)

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
    from wrappers_mine import TimeLimit
    import matplotlib.pyplot as plt
    rng = jax.random.PRNGKey(0)

    env = GridEnv(4)
    env = TimeLimit(env, 5)

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


def test_rew():
    from wrappers_mine import TimeLimit
    from wrappers import LogWrapper
    import matplotlib.pyplot as plt
    rng = jax.random.PRNGKey(0)

    env = GridEnv(8)
    env = TimeLimit(env, 128)
    env = LogWrapper(env)

    rng, _rng = jax.random.split(rng)
    env_params = env.sample_params(_rng)

    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng, env_params)

    a = []
    for i in range(256):
        rng, _rng = jax.random.split(rng)
        act = jax.random.randint(_rng, (), 0, env.n_acts)
        rng, _rng = jax.random.split(rng)
        obs, state, rew, done, info = env.step(_rng, state, act, env_params)
        # a.append(rew)
        a.append(info['returned_episode_returns'])
    a = jnp.stack(a)
    print(a)
    # print(info)


if __name__ == '__main__':
    test_rew()
