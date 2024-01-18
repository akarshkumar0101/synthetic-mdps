import pickle
from functools import partial

import distrax
import jax
import jax.config as jax_config
import jax.numpy as jnp
import optax
from einops import rearrange
from flax.training.train_state import TrainState
from jax.random import split
from tqdm.auto import tqdm

from agents import DenseObsEmbed, ObsActRewTimeEmbed, LinearTransformerAgent

jax_config.update("jax_debug_nans", True)


def single_episode(rng, gridlen=32, n_trials=4, n_steps=32):
    act_map = jnp.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    rng, _rng = split(rng)
    state_goal = jax.random.randint(_rng, (2,), 0, gridlen)
    rng, _rng = split(rng)
    # _rng = jax.random.PRNGKey(0)  # TODO: fixed obs_matrix
    obs_matrix = jax.random.normal(_rng, (4, 2))

    def single_trial(rng):
        def step(carry, _):
            rng, state, = carry

            delta = (state_goal - state).clip(-1, 1)
            logits = 1.5 * jnp.array([delta[0] > 0, delta[1] > 0, delta[0] < 0, delta[1] < 0]) * 100
            rng, _rng = split(rng)
            act = jax.random.categorical(_rng, logits)

            obs = obs_matrix @ (state / gridlen * 2. - 1.)

            state_n = state + act_map[act]
            carry = rng, state_n,
            return carry, dict(obs=obs, state=state, logits=logits, act=act)

        rng, _rng = split(rng)
        state_start = jax.random.randint(_rng, (2,), 0, gridlen)
        carry = rng, state_start,
        carry, buffer = jax.lax.scan(step, carry, jnp.arange(n_steps))
        return buffer

    return jax.vmap(single_trial)(split(rng, n_trials))


def kl_div(logits, logits_target, axis=-1):
    log_p, log_q = jax.nn.log_softmax(logits_target), jax.nn.log_softmax(logits)
    return (jnp.exp(log_p) * (log_p - log_q)).sum(axis=axis)


def main():
    print('2')
    n_envs, n_steps = 32, 128
    n_acts = 4

    def loss_fn(agent_params, rng):
        buffer = jax.vmap(single_episode)(split(rng, n_envs))
        buffer = jax.tree_map(lambda x: rearrange(x, 'B K T ... -> B (K T) ...'), buffer)

        agent_state, _ = jax.vmap(init_with_out_init_state, in_axes=(None, 0))(rng, split(_rng, n_envs))

        obs = dict(obs=buffer['obs'], act_p=jnp.zeros((n_envs, n_steps), dtype=int), rew_p=jnp.zeros((n_envs, n_steps)),
                   done=jnp.zeros((n_envs, n_steps), dtype=bool))
        forward_parallel = partial(agent.apply, method=agent.forward_parallel)
        _, (logits, val) = jax.vmap(forward_parallel, in_axes=(None, 0, 0))(agent_params, agent_state, obs)
        entropy = distrax.Categorical(logits=logits).entropy()
        # loss = kl_div(logits, buffer['logits'], axis=-1).mean()
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, buffer['act'])
        return loss.mean(), (loss, entropy)

    rng = jax.random.PRNGKey(0)

    ObsEmbed = partial(DenseObsEmbed, d_embd=128)
    ObsEmbed = partial(ObsActRewTimeEmbed, d_embd=128, ObsEmbed=ObsEmbed, n_acts=n_acts, n_steps_max=500)
    agent = LinearTransformerAgent(ObsEmbed, n_acts=n_acts, n_layers=4, n_heads=4, d_embd=128)

    init_with_out_init_state = partial(agent.init_with_output, method=agent.init_state)
    rng, _rng = split(rng)
    agent_state, _ = jax.vmap(init_with_out_init_state, in_axes=(None, 0))(rng, split(_rng, n_envs))

    agent_state0 = jax.tree_map(lambda x: x[0], agent_state)
    obs0 = dict(obs=jnp.zeros((4,)), act_p=jnp.zeros((), dtype=int), rew_p=jnp.zeros(()),
                done=jnp.zeros((), dtype=bool))
    rng, _rng = split(rng)
    agent_params = agent.init(_rng, agent_state0, obs0, method=agent.forward_recurrent)

    tx = optax.chain(optax.clip_by_global_norm(1.), optax.adam(3e-4, eps=1e-5))
    train_state = TrainState.create(apply_fn=None, params=agent_params, tx=tx)

    pbar = tqdm(range(10000))
    for i in pbar:
        rng = jax.random.PRNGKey(i)
        (loss_avg, (loss, entropy)), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params, rng=rng)
        train_state = train_state.apply_gradients(grads=grads)

        a = loss.mean(axis=0).reshape(4, 32)
        pbar.set_postfix(loss=loss_avg.item(), ppl=jnp.exp(entropy.mean()).item(),
                         loss_trial_0=a[0].mean().item(), loss_trial_3=a[3].mean().item())
        if i % 100 == 0:
            with open(f'../data/grid_train_agent_params.pkl', 'wb') as f:
                pickle.dump(train_state.params, f)


if __name__ == '__main__':
    main()
