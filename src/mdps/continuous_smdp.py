import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces
import flax.linen as nn


class ContinuousInit(nn.Module):
    d_state: int
    initializer: nn.initializers.Initializer = nn.initializers.normal(stddev=1.)
    normalize: bool = True

    def setup(self):
        pass

    def __call__(self, rng):
        state = self.initializer(rng, (self.d_state,))
        if self.normalize:
            state = state / jnp.linalg.norm(state)
        return state


class ContinuousMatrixTransition(nn.Module):
    d_state: int
    initializer: nn.initializers.Initializer = nn.initializers.normal(stddev=1.)
    use_bias: bool = False
    normalize: bool = True

    def setup(self):
        self.trans_matrix = nn.Dense(self.d_state, kernel_init=self.initializer, use_bias=self.use_bias)

    def __call__(self, state, rng):
        state_n = state+self.trans_matrix(state)
        if self.normalize:
            state_n = state_n / jnp.linalg.norm(state_n)
        return state_n


class ContinuousMatrixObs(nn.Module):
    d_state: int
    d_obs: int
    initializer: nn.initializers.Initializer = nn.initializers.normal(stddev=1.)
    use_bias: bool = False

    def setup(self):
        self.obs_matrix = nn.Dense(self.d_obs, kernel_init=self.initializer, use_bias=self.use_bias)

    def __call__(self, state):
        return self.obs_matrix(state)

    def observation_space(self, params):
        return spaces.Box(-1, 1, (self.d_obs,), dtype=jnp.float32)


class ContinuousReward(nn.Module):
    d_state: int
    initializer: nn.initializers.Initializer = nn.initializers.normal(stddev=1.)
    normalize: int = True

    def setup(self):
        self.rew_vec = self.param('rew_vec', self.initializer, (self.d_state,))
        if self.normalize:
            self.rew_vec = self.rew_vec / jnp.linalg.norm(self.rew_vec)

    def __call__(self, state):
        rew = jnp.dot(state, self.rew_vec)
        return (rew+1)/2.


def main():
    from syntheticmdp import SyntheticMDP
    rng = jax.random.PRNGKey(0)

    d_state, d_obs = 32, 10
    n_acts = 4

    model_init = ContinuousInit(d_state)
    model_trans = ContinuousMatrixTransition(d_state)
    model_obs = ContinuousMatrixObs(d_state, d_obs)
    model_rew = ContinuousReward(d_state)
    env = SyntheticMDP(None, None, n_acts, model_init, model_trans, model_obs, model_rew)

    rng, _rng = jax.random.split(rng)
    env_params = env.sample_params(_rng)

    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng, env_params)

    for t in range(20):
        rng, _rng = jax.random.split(rng)
        act = jax.random.randint(_rng, (), 0, n_acts)
        rng, _rng = jax.random.split(rng)
        obs, state, rew, done, info = env.step(_rng, state, act, env_params)
        print(rew)


if __name__ == "__main__":
    main()
