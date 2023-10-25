import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces
import flax.linen as nn



class DiscreteInit(nn.Module):
    n_states: int
    initializer: nn.initializers.Initializer = nn.initializers.normal(stddev=1.)

    def setup(self):
        self.logits = self.param('logits', self.initializer, (self.n_states,))

    def __call__(self, rng):
        return jax.random.categorical(rng, self.logits)


class DiscreteTransition(nn.Module):
    n_states: int
    initializer: nn.initializers.Initializer = nn.initializers.normal(stddev=1.)

    def setup(self):
        self.trans_matrix = self.param('trans_matrix', self.initializer, (self.n_states, self.n_states))

    def __call__(self, state, rng):
        logits = self.trans_matrix[:, state]
        state_n = jax.random.categorical(rng, logits)
        return state_n


class DiscreteObs(nn.Module):
    n_states: int
    d_obs: int
    initializer: nn.initializers.Initializer = nn.initializers.normal(stddev=1.)

    def setup(self):
        self.embed = nn.Embed(self.n_states, self.d_obs, embedding_init=self.initializer)

    def __call__(self, state):
        return self.embed(state)


class DiscreteReward(nn.Module):
    n_states: int
    initializer: nn.initializers.Initializer = nn.initializers.normal(stddev=1.)

    def setup(self):
        self.rew_matrix = self.param('rew_matrix', self.initializer, (self.n_states,))

    def __call__(self, state):
        return self.rew_matrix[state]


def main():
    from wrappers_mine import TimeLimit
    from syntheticmdp import SyntheticMDP
    from matplotlib import pyplot as plt
    rng = jax.random.PRNGKey(0)

    n_states, d_obs = 64, 10
    n_acts = 4

    model_init = DiscreteInit(n_states)
    model_trans = DiscreteTransition(n_states, initializer=nn.initializers.normal(stddev=100))
    model_obs = DiscreteObs(n_states, d_obs)
    model_rew = DiscreteReward(n_states)
    env = SyntheticMDP(None, None, n_acts, model_init, model_trans, model_obs, model_rew)
    env = TimeLimit(env, 4)

    rng, _rng = jax.random.split(rng)
    env_params = env.sample_params(_rng)

    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng, env_params)

    for j in range(5):
        print()
        print()
        print()
        for i in range(4):
            rng, _rng = jax.random.split(rng)
            act = jax.random.randint(_rng, (), 0, n_acts)
            rng, _rng = jax.random.split(rng)
            obs, state, rew, done, info = env.step(_rng, state, act, env_params)
            print(done)

    # rng, _rng = jax.random.split(rng)
    # env_params = env.sample_params(_rng)
    # print(jax.tree_map(lambda x: x.shape, env_params))
    #
    # rng, _rng = jax.random.split(rng)
    # obs, state = env.reset(_rng, env_params)
    #
    # for i in range(100):
    #     rng, _rng = jax.random.split(rng)
    #     act = jax.random.randint(_rng, (), 0, n_acts)
    #     rng, _rng = jax.random.split(rng)
    #     obs, state, rew, done, info = env.step(_rng, state, act, env_params)
    #
    # trans_matrix = env_params[1]['params']['trans_matrix'][0]
    # from einops import rearrange
    # a = rearrange(env_params[1]['params']['trans_matrix'], 'a h w -> h w a')[:, :, :3]
    # plt.imshow(jax.nn.softmax(a, axis=0))
    # plt.show()


if __name__ == "__main__":
    main()


