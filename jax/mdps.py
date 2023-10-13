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
        state = self.get_transition(state, action)
        obs, rew = self.get_obs_rew(state)
        return state, obs, rew

    def get_transition(self, state, action):
        state = self.transition_fn(state)
        state = rearrange(state, '(a d) -> a d', a=self.n_actions)
        state = state[0, :]
        return state

    def get_obs_rew(self, state):
        obs = self.obs_fn(state)
        rew = self.reward_fn(state)[..., 0]
        return obs, rew


def main():
    from functools import partial
    rng = jax.random.PRNGKey(0)

    bs = 3
    n_steps = 32
    n_actions = 10
    d_state = 8
    d_obs = 32
    d_embd = 128

    from agent import Transformer
    net = Transformer(n_actions=n_actions, n_steps=n_steps, n_layers=3, n_heads=4, d_embd=d_embd)
    obs = jnp.zeros((bs, n_steps, d_obs))
    action = jnp.zeros((bs, n_steps), dtype=jnp.int32)
    reward = jnp.zeros((bs, n_steps))
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

    obs = jnp.zeros((bs, d_obs))
    action = jnp.zeros((bs,), dtype=jnp.int32)
    reward = jnp.zeros((bs,))
    time = jnp.zeros((bs,), dtype=jnp.int32)
    agent_state = net.get_init_state(bs)

    print(env_state.shape, obs.shape, action.shape, reward.shape, time.shape)
    print(jax.tree_map(lambda x: x.shape, agent_state))

    forward_parallel = jax.vmap(partial(net.apply, params, method=net.forward_parallel))
    forward_recurrent = jax.vmap(partial(net.apply, params, method=net.forward_recurrent))
    get_transition = partial(jax.vmap(partial(mdp.apply, method=mdp.get_transition)), params_mdp)
    get_obs_rew = partial(jax.vmap(partial(mdp.apply, method=mdp.get_obs_rew)), params_mdp)

    # for i in range(10):
    #     agent_state, (logits, values) = forward_recurrent(agent_state, (obs, action, reward, time))
    #     rng, _rng = jax.random.split(rng)
    #     action = jax.random.categorical(_rng, logits=logits)
    #     env_state, (obs, reward) = jax.vmap(mdp.apply)(params_mdp, env_state, action)

    def collect_step(carry, x):
        rng, agent_state_t, env_state_t, (act_tp, rew_tp, time_tp) = carry
        time_t = time_tp + 1
        obs_t, rew_t = get_obs_rew(env_state_t)

        agent_state_tn, (logits_t, val_t) = forward_recurrent(agent_state_t, (obs_t, act_tp, rew_tp, time_t))
        rng, _rng = jax.random.split(rng)
        act_t = jax.random.categorical(_rng, logits=logits_t)
        env_state_tn = get_transition(env_state_t, act_t)

        carry = rng, agent_state_tn, env_state_tn, (act_t, rew_t, time_t)
        return carry, dict(env_state=env_state_t, obs=obs_t, act_tp=act_tp, act=act_t, rew_tp=rew_tp, rew=rew_t, time=time_t, logits=logits_t, val=val_t)

    carry = rng, agent_state, env_state, (action, reward, time)
    carry, buffer = jax.lax.scan(collect_step, carry, jnp.arange(30))
    print(jax.tree_map(lambda x: x.shape, carry))

    print(jax.tree_map(lambda x: x.shape, buffer))
    buffer = jax.tree_map(lambda x: rearrange(x, 't b ... -> b t ...'), buffer)
    print(jax.tree_map(lambda x: x.shape, buffer))

    logits, vals = forward_parallel(buffer['obs'], buffer['act_tp'], buffer['rew_tp'], buffer['time'])
    print(vals.shape)
    print(buffer['val'].shape)

    print(jnp.allclose(vals, buffer['val'], atol=1e-3))
    print(jnp.allclose(logits, buffer['logits'], atol=1e-3))




if __name__ == '__main__':
    main()

"""

----- WHAT THE BUFFER IS -----
observation: o0, o1, o2
    actions: a0, a1, a2
    rewards: r0, r1, r2
      times: t0, t1, t2

----- WHAT THE MODEL SEES -----

 prediction: a0, a1, a2
observation: o0, o1, o2
    actions: __, a0, a1, a2
    rewards: __, r0, r1, r2
      times: t0, t1, t2
"""
