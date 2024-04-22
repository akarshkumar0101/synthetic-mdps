import jax
import jax.numpy as jnp
from einops import rearrange, repeat
from functools import partial

from linear_transformer import LinearTransformerAgent
from basic import BasicAgent

def main():
    from functools import partial

    rng = jax.random.PRNGKey(0)
    bs, n_steps, d = 8, 32, 128
    n_acts = 4

    net = LinearTransformerAgent(n_acts=n_acts, n_steps=n_steps, n_layers=3, n_heads=4, d_embd=128)
    # net = BasicAgent(n_acts=n_acts)
    forward_parallel = jax.vmap(partial(net.apply, method=net.forward_parallel), in_axes=(None, 0, 0, 0))
    forward_recurrent = jax.vmap(partial(net.apply, method=net.forward_recurrent), in_axes=(None, 0, (0, 0, 0)))

    # ------ INIT ------
    rng, *_rng = jax.random.split(rng, 1 + 4)
    obs = jax.random.normal(_rng[0], (bs, n_steps, 64))
    action = jax.random.randint(_rng[1], (bs, n_steps,), 0, n_acts)
    reward = jax.random.normal(_rng[2], (bs, n_steps,))

    rng, _rng = jax.random.split(rng)
    _rng = jax.random.split(_rng, bs)
    agent_state = jax.vmap(net.get_init_state)(_rng)
    print('agent_state shape')
    print(jax.tree_map(lambda x: x.shape, agent_state))

    rng, _rng = jax.random.split(rng)
    params = net.init(_rng, obs[0], action[0], reward[0], method=net.forward_parallel)

    print('params shape')
    print(jax.tree_map(lambda x: x.shape, params))

    # ------ PARALLEL FORWARD ------
    logits1, values1 = forward_parallel(params, obs, action, reward)

    # ------ RECURRENT FORWARD ------
    oar_ = jax.tree_map(lambda x: rearrange(x, 'b t ... -> t b ...'), (obs, action, reward))
    state, (logits2, values2) = jax.lax.scan(partial(forward_recurrent, params), agent_state, oar_)
    logits2 = rearrange(logits2, 't b ... -> b t ...')
    values2 = rearrange(values2, 't b ... -> b t ...')

    assert jnp.allclose(logits1, logits2, atol=1e-3)
    assert jnp.allclose(values1, values2, atol=1e-3)


if __name__ == '__main__':
    main()
