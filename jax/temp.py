import jax
import jax.numpy as jnp


def rollout(agent, env, rng, agent_params, env_params, agent_state, env_state, oart, n_steps):
    def rollout_step(carry, _):
        rng, agent_state, env_state, oart = carry
        obs, act_p, rew_p, time = oart
        rng, _rng_agent, _rng_env = jax.random.split(rng, 3)
        agent_state, (logits, val) = agent.apply(agent_params, agent_state, (obs, act_p, rew_p, time),
                                                 method=agent.forward_recurrent)
        act = jax.random.categorical(_rng_agent, logits=logits).astype(jnp.int32)
        obs_n, env_state, rew, done, info = env.step(_rng_env, env_state, act, env_params)
        oart = obs_n, act, rew, time + 1
        carry = rng, agent_state, env_state, oart
        return carry, dict(obs=obs, act=act, rew=rew, done=done,
                           time=time, logits=logits, val=val, act_p=act_p, rew_p=rew_p, info=info)

    carry = rng, agent_state, env_state, oart
    carry, buffer = jax.lax.scan(rollout_step, carry, jnp.arange(n_steps))
    return carry, buffer


def calc_gae(buffer, val_last, gamma=0.99, gae_lambda=0.95):
    def gae_step(carry, transition):
        gae, next_value = carry
        done, value, reward = transition['done'], transition['val'], transition['rew']
        delta = reward + gamma * next_value * (1 - done) - value
        gae = (delta + gamma * gae_lambda * (1 - done) * gae)
        return (gae, value), gae

    carry = jnp.zeros_like(val_last), val_last
    _, adv = jax.lax.scan(gae_step, carry, buffer, reverse=True, unroll=16)
    ret = adv + buffer['val']
    return adv, ret


