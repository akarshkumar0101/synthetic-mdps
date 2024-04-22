from functools import partial

import gymnax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from einop import einop
from jax.random import split
from tqdm.auto import tqdm

from agents.basic import BasicAgent
from agents.util import DenseObsEmbed
from algos.ppo_dr import PPO
from wrappers import LogWrapper


def main():
    rng = jax.random.PRNGKey(0)
    env, env_params = gymnax.make("CartPole-v1")
    n_acts = env.action_space(env_params).n
    env = LogWrapper(env)
    agent = BasicAgent(ObsEmbed=partial(DenseObsEmbed, 64), n_acts=n_acts)

    ppo = PPO(agent, env, sample_env_params=lambda rng: env.default_params)
    init_agent_env = jax.vmap(ppo.init_agent_env)
    eval_step = jax.jit(jax.vmap(ppo.eval_step, in_axes=(0, None)))
    ppo_step = jax.jit(jax.vmap(ppo.ppo_step, in_axes=(0, None)))

    rng, _rng = split(rng)
    carry = init_agent_env(split(_rng, 32))
    rets = []
    pbar = tqdm(range(500))
    for _ in pbar:
        carry, buffer = ppo_step(carry, None)
        rets.append(buffer['info']['returned_episode_returns'])
        pbar.set_postfix(ret=rets[-1].mean())
    rets = jnp.stack(rets, axis=1)
    print(rets.shape)

    steps = jnp.arange(rets.shape[1]) * 128 * 4
    plt.plot(steps, einop(rets, 's n t e -> n', reduction='mean'), label='mean')
    plt.plot(steps, einop(rets, 's n t e -> n', reduction=jnp.median), label='median')
    plt.plot(steps, einop(rets, 's n t e -> n s', reduction='mean'), c='gray', alpha=0.1)
    plt.legend()
    plt.ylabel('Return')
    plt.xlabel('Env Steps')
    plt.show()
    print('done')


if __name__ == '__main__':
    main()
