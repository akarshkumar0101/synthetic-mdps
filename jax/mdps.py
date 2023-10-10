import jax
import jax.numpy as jnp
from flax import linen as nn
from einops import rearrange, repeat
import math


class LinearSyntheticMDP(nn.Module):
    n_actions: int
    d_state: int
    d_obs: int

    def setup(self):
        self.transition_fn = nn.Dense(features=self.n_actions * self.d_state, use_bias=False)
        self.obs_fn = nn.Dense(features=self.d_obs, use_bias=False)
        self.reward_fn = nn.Dense(features=1, use_bias=False)

    def __call__(self, state, action):
        state = self.transition_fn(state)
        state = rearrange(state, '... (a d) -> ... a d', a=self.n_actions)
        # state = jnp.take_along_axis(state, action[:, None, None], axis=1)
        state = state[..., 0, :]
        obs = self.obs_fn(state)
        reward = self.reward_fn(state)
        return state, (obs, reward)


def main():
    rng = jax.random.PRNGKey(0)

    mdp = LinearSyntheticMDP(n_actions=5, d_state=8, d_obs=32)
    state = jnp.zeros((1, 8))
    action = jnp.zeros((1, ), dtype=jnp.int32)

    params = mdp.init(rng, state, action)

    print(jax.tree_map(lambda x: x.shape, params))


def main():
    rng = jax.random.PRNGKey(0)

    bs = 8
    n_steps = 32
    n_actions = 10
    d_state = 8
    d_obs = 32
    d_embd = 128

    from agent import Transformer
    net = Transformer(n_actions=n_actions, n_steps=n_steps, n_layers=3, n_heads=4, d_embd=d_embd)
    obs = jnp.zeros((bs, n_steps, d_obs))
    action = jnp.zeros((bs, n_steps), dtype=jnp.int32)
    reward = jnp.zeros((bs, n_steps, 1))
    time = jnp.zeros((bs, n_steps), dtype=jnp.int32)
    params = net.init(rng, obs, action, reward, time)
    print(jax.tree_map(lambda x: x.shape, params))

    mdp = LinearSyntheticMDP(n_actions=n_actions, d_state=d_state, d_obs=d_obs)
    # env_state = jnp.zeros((1, 8, ))
    # action = jnp.zeros((1, ), dtype=jnp.int32)
    # params_mdp = mdp.init(rng, env_state, action)

    env_state = jnp.zeros((bs, d_state))
    action = jnp.zeros((bs, ), dtype=jnp.int32)
    rng, *_rng = jax.random.split(rng, 1+bs)
    params_mdp = jax.vmap(mdp.init)(jnp.stack(_rng), env_state, action)

    print(jax.tree_map(lambda x: x.shape, params_mdp))

    env_state, (o, r) = jax.vmap(mdp.apply)(params_mdp, env_state, action)

    env_state = jnp.zeros((bs, d_state))
    obs = jnp.zeros((bs, d_obs))
    agent_state = net.get_init_state(bs)

    action = jnp.zeros((bs, ), dtype=jnp.int32)
    reward = jnp.zeros((bs, 1))
    time = jnp.zeros((bs, ), dtype=jnp.int32)
    net.apply(params, obs, action, reward, time)

    # for i in range(1):
    #     net.apply(params, obs, action, reward, time)
    #     agent_state, (logits, values) = net.forward_recurrent(agent_state, (o, action, r, time))
    #     print(logits.shape, values.shape)





if __name__ == '__main__':
    main()
