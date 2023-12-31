from functools import partial

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import linen as nn
from jax.random import split


def linear_attention(state, qkv):
    # state: D, D
    q, k, v = qkv  # T, D
    T, D = q.shape
    mask = jnp.tril(jnp.ones((T, T), dtype=bool))  # causal masking
    attn = mask * (q @ k.T)
    out = attn @ v + q @ state  # (T, T) @ (T, D) + (T, D) @ (D, D) = (T, D)
    state = state + k.T @ v  # (D, D) + (D, T) @ (T, D) = (D, D)
    return state, out


class MultiHeadAttention(nn.Module):
    n_heads: int
    d_embd: int

    def setup(self):
        self.lin_qkv = nn.Dense(features=3 * self.d_embd)
        self.lin_out = nn.Dense(features=self.d_embd)

    def __call__(self, state, x):
        T, D = x.shape
        assert D == self.d_embd
        qkv = self.lin_qkv(x)  # (T, 3 * D)
        q, k, v = rearrange(qkv, 'T (QKV H Dh) -> QKV H T Dh', QKV=3, H=self.n_heads)  # (H, T, Dh)
        state, x = jax.vmap(linear_attention)(state, (q, k, v))  # (H, T, Dh)
        x = rearrange(x, 'H T Dh -> T (H Dh)')  # (T, D)
        x = self.lin_out(x)  # (T, D)
        return state, x


class MLP(nn.Module):
    d_embd: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=4 * self.d_embd)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=self.d_embd)(x)
        return x


class Block(nn.Module):
    n_heads: int
    d_embd: int

    def setup(self):
        self.mha = MultiHeadAttention(n_heads=self.n_heads, d_embd=self.d_embd)
        self.ln1 = nn.LayerNorm()
        self.mlp = MLP(d_embd=self.d_embd)
        self.ln2 = nn.LayerNorm()

    def __call__(self, state, x):
        state, dx = self.mha(state, self.ln1(x))
        x = x + dx
        x = x + self.mlp(self.ln2(x))
        return state, x

    # def initialize_carry(self, rng):
    #     d_head = self.d_embd // self.n_heads
    #     return jnp.zeros((self.n_heads, d_head, d_head))


class ObsActRewTimeEmbed(nn.Module):
    d_embd: int
    n_acts: int
    n_steps_max: int

    @nn.compact
    def __call__(self, state, x):
        # state: (), x: (T, ...)
        obs, act_p, rew_p = x['obs'], x['act_p'], x['rew_p']
        T, = act_p.shape
        time = state + jnp.arange(T)

        x_obs = nn.Dense(features=self.d_embd)(obs)  # T, D
        x_act = nn.Embed(self.n_acts, features=self.d_embd)(act_p)  # T, D
        x_rew = nn.Dense(features=self.d_embd)(rew_p[..., None])  # T, D
        x_time = nn.Embed(self.n_steps_max, features=self.d_embd)(time)  # T, D
        x = x_obs + x_act + x_rew + x_time
        state = state + T
        return state, x

    # def initialize_carry(self, rng):
    #     return jnp.zeros((), dtype=jnp.int32)


class LinearTransformerAgent(nn.Module):
    n_acts: int
    n_steps_max: int
    n_layers: int
    n_heads: int
    d_embd: int

    def setup(self):
        self.embed_obs = ObsActRewTimeEmbed(d_embd=self.d_embd, n_acts=self.n_acts, n_steps_max=self.n_steps_max)
        self.blocks = [Block(n_heads=self.n_heads, d_embd=self.d_embd) for _ in range(self.n_layers)]
        self.ln = nn.LayerNorm()
        self.actor = nn.Dense(features=self.n_acts, kernel_init=nn.initializers.orthogonal(0.01),
                              bias_init=nn.initializers.constant(0.0))  # T, A
        self.critic = nn.Dense(features=1)  # T, 1

    @nn.compact
    def __call__(self, state, x):
        state_obs, state_blocks = state['state_obs'], state['state_blocks']
        state_obs, x = self.embed_obs(state_obs, x)

        state_blocks_out = [None] * self.n_layers
        for i_layer in range(self.n_layers):
            state_blocks_out[i_layer], x = self.blocks[i_layer](state_blocks[i_layer], x)

        x = self.ln(x)
        logits, val = self.actor(x), self.critic(x)  # (T, A) and (T, 1)
        state = dict(state_obs=state_obs, state_blocks=state_blocks_out)
        return state, (logits, val[..., 0])

    def initialize_carry(self, rng):
        d_head = self.d_embd // self.n_heads
        state_obs = jnp.zeros((), dtype=jnp.int32)
        state_blocks = [jnp.zeros((self.n_heads, d_head, d_head)) for _ in range(self.n_layers)]
        return dict(state_obs=state_obs, state_blocks=state_blocks)

    # def initialize_carry(self, rng):
    #     rng, _rng = split(rng)
    #     state_obs = ObsActRewTimeEmbed.initialize_carry(_rng)
    #     rng, *_rng = split(rng, 1 + self.n_layers)
    #     state_blocks = [Block.initialize_carry(_rng) for block, _rng in zip(self.blocks, _rng)]
    #     return dict(state_obs=state_obs, state_blocks=state_blocks)


def main():
    rng = jax.random.PRNGKey(0)
    B, T, D = 8, 32, 128
    n_acts = 4

    agent = LinearTransformerAgent(n_acts=n_acts, n_steps_max=T, n_layers=3, n_heads=4, d_embd=128)

    rng, _rng = split(rng)
    agent_state = jax.vmap(agent.initialize_carry)(split(_rng, B))

    # ------ INIT ------
    rng, *_rng = split(rng, 1 + 3)
    obs = jax.random.normal(_rng[0], (B, T, 64))
    act = jax.random.randint(_rng[1], (B, T,), 0, n_acts)
    rew = jax.random.normal(_rng[2], (B, T,))

    obs = dict(obs=obs, act_p=act, rew_p=rew)

    rng, _rng = split(rng)
    obs0 = jax.tree_map(lambda x: x[0], obs)
    agent_state0 = jax.tree_map(lambda x: x[0], agent_state)
    params = agent.init(_rng, agent_state0, obs0)
    forward = jax.vmap(partial(agent.apply, params))

    print('agent_state.shape: ', jax.tree_map(lambda x: x.shape, agent_state))
    print('params.shape: ', jax.tree_map(lambda x: x.shape, params))
    print('obs.shape: ', jax.tree_map(lambda x: x.shape, obs))

    # ------ PARALLEL FORWARD ------
    _, out1 = forward(agent_state, obs)
    print('out1.shape: ', jax.tree_map(lambda x: x.shape, out1))

    # ------ RECURRENT FORWARD ------
    a = jax.tree_map(lambda x: rearrange(x, 'b t ... -> t b 1 ...'), obs)
    _, out2 = jax.lax.scan(forward, agent_state, a)
    out2 = jax.tree_map(lambda x: rearrange(x, 't b 1 ... -> b t ...'), out2)
    print(jax.tree_map(lambda x, y: jnp.allclose(x, y, atol=1e-4).item(), out1, out2))

    # state, (logits2, values2) = jax.lax.scan(partial(forward_recurrent, params), agent_state, obs_)
    # logits2 = rearrange(logits2, 't b ... -> b t ...')
    # values2 = rearrange(values2, 't b ... -> b t ...')
    #
    # assert jnp.allclose(logits1, logits2, atol=1e-3)
    # assert jnp.allclose(values1, values2, atol=1e-3)


def temp_test():
    rng = jax.random.PRNGKey(0)
    q, k, v = jax.random.normal(rng, (3, 200, 32))
    state = jnp.zeros((32, 32))

    _, out1 = linear_attention(state, (q, k, v))

    q, k, v = jax.tree_map(lambda x: x.reshape(20, 10, 32), (q, k, v))
    _, out2 = jax.lax.scan(linear_attention, state, (q, k, v))
    out2 = out2.reshape(200, 32)
    print(jnp.allclose(out1, out2, atol=1e-4))


if __name__ == '__main__':
    main()
