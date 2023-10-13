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
        state = rearrange(state, '(a d) -> a d', a=self.n_actions)
        # state = jnp.take_along_axis(state, action[:, None, None], axis=1)
        state = state[0, :]
        obs = self.obs_fn(state)
        reward = self.reward_fn(state)
        return state, (obs, reward)


def main():
    from functools import partial
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

    rng, _rng = jax.random.split(rng)
    params = net.init(_rng, obs[0], action[0], reward[0], time[0])
    print(jax.tree_map(lambda x: x.shape, params))

    mdp = LinearSyntheticMDP(n_actions=n_actions, d_state=d_state, d_obs=d_obs)
    env_state = jnp.zeros((bs, d_state))
    action = jnp.zeros((bs,), dtype=jnp.int32)
    rng, *_rng = jax.random.split(rng, 1 + bs)
    params_mdp = jax.vmap(mdp.init)(jnp.stack(_rng), env_state, action)
    print(jax.tree_map(lambda x: x.shape, params_mdp))

    # ------------------------------------------------------------------------
    env_state = jnp.zeros((bs, d_state))

    obs = jnp.zeros((bs, 1, d_obs))
    action = jnp.zeros((bs, 1,), dtype=jnp.int32)
    reward = jnp.zeros((bs, 1, 1))
    time = jnp.zeros((bs, 1,), dtype=jnp.int32)
    agent_state = net.get_init_state(bs)

    print(env_state.shape, obs.shape, action.shape, reward.shape, time.shape)
    print(jax.tree_map(lambda x: x.shape, agent_state))

    forward_recurrent = partial(net.apply, params, method=net.forward_recurrent)
    forward_recurrent = jax.vmap(forward_recurrent, in_axes=(0, (0, 0, 0, 0)))

    for i in range(10):
        agent_state, (logits, values) = forward_recurrent(agent_state, (obs, action, reward, time))
        rng, _rng = jax.random.split(rng)
        action = jax.random.categorical(_rng, logits=logits)
        env_state, (obs, reward) = jax.vmap(mdp.apply)(params_mdp, env_state, action)
        obs = obs[:, None, :]
        reward = reward[:, None, :]



    # jax.lax.scan()


if __name__ == '__main__':
    main()
