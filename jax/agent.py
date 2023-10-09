import jax
import jax.numpy as jnp
from flax import linen as nn

from einops import rearrange, repeat

import math


def ret_parallel_forward(qkv):
    q, k, v = qkv
    # q, k, v: B, H, T, D
    B, H, T, D = q.shape
    mask = jnp.tril(jnp.ones((T, T), dtype=bool))  # causal masking
    attn = mask * (q @ k.mT)
    out = attn @ v
    return out


def ret_recurrent_forward(state, qkv):
    q, k, v = qkv
    # state: B, H, D, D
    # q, k, v: B, H, 1, D
    B, H, T, D = q.shape
    assert T == 1
    state = state + k.mT * v  # B, H, D, D
    out = q @ state  # B, H, 1, D
    return state, out  # state: B, H, D, D; out: B, H, 1, D


class MultiHeadAttention(nn.Module):
    n_heads: int
    d_embd: int

    def setup(self):
        assert self.d_embd % self.n_heads == 0
        self.d_head = self.d_embd // self.n_heads

        self.dense1 = nn.Dense(features=3 * self.d_embd)
        self.dense2 = nn.Dense(features=self.d_embd)

    def __call__(self, x, mask=None):
        bs, t, d_embd = x.shape
        d_head = d_embd // self.n_heads
        if mask is None:  # assume causal mask
            mask = jnp.tril(jnp.ones((t, t), dtype=bool))
            mask = repeat(mask, 't t2 -> b h t t2', b=bs, h=self.n_heads)
        kqv = self.dense1(x)  # (bs, t, 3 * d_embd)
        k, q, v = jnp.split(kqv, 3, axis=-1)  # (bs, t, d_embd)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.n_heads)  # (bs, n_heads, t, d_head)
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.n_heads)  # (bs, n_heads, t, d_head)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.n_heads)  # (bs, n_heads, t, d_head)
        attn = (q @ k.mT) * (1.0 / math.sqrt(d_head))  # (bs, n_heads, t, t)
        attn = jax.lax.select(mask, attn, jnp.full_like(attn, -float("inf")))
        # attn = nn.softmax(attn, axis=-1) # (bs, n_heads, t, t)
        x = attn @ v  # (bs, n_heads, t, d_head)
        x = rearrange(x, 'b h t d -> b t (h d)')  # (bs, t, d_embd)
        x = self.dense2(x)  # (bs, t, d_embd)
        return x

    def forward_parallel(self, x):
        bs, t, d_embd = x.shape
        assert d_embd == self.d_embd
        kqv = self.dense1(x)  # (bs, t, 3 * d_embd)
        k, q, v = jnp.split(kqv, 3, axis=-1)  # (bs, t, d_embd)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.n_heads)  # (bs, n_heads, t, d_head)
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.n_heads)  # (bs, n_heads, t, d_head)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.n_heads)  # (bs, n_heads, t, d_head)
        x = ret_parallel_forward((q, k, v))  # (bs, n_heads, t, d_head)
        x = rearrange(x, 'b h t d -> b t (h d)')  # (bs, t, d_embd)
        x = self.dense2(x)  # (bs, t, d_embd)
        return x

    def forward_recurrent(self, state, x):
        bs, t, d_embd = x.shape
        assert d_embd == self.d_embd
        assert t == 1
        kqv = self.dense1(x)  # (bs, t, 3 * d_embd)
        k, q, v = jnp.split(kqv, 3, axis=-1)  # (bs, t, d_embd)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.n_heads)  # (bs, n_heads, t, d_head)
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.n_heads)  # (bs, n_heads, t, d_head)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.n_heads)  # (bs, n_heads, t, d_head)
        state, x = ret_recurrent_forward(state, (q, k, v))  # (bs, n_heads, d_head, d_head), (bs, n_heads, 1, d_head)
        x = rearrange(x, 'b h 1 d -> b 1 (h d)')  # (bs, t, d_embd)
        x = self.dense2(x)  # (bs, t, d_embd)
        return state, x


class MLP(nn.Module):
    d_embd: int

    def setup(self):
        self.dense1 = nn.Dense(features=4 * self.d_embd)
        self.dense2 = nn.Dense(features=self.d_embd)

    def __call__(self, x):
        x = self.dense1(x)
        x = nn.gelu(x)
        x = self.dense2(x)
        return x


class Block(nn.Module):
    n_heads: int
    d_embd: int

    def setup(self):
        self.mha = MultiHeadAttention(n_heads=self.n_heads, d_embd=self.d_embd)
        self.ln1 = nn.LayerNorm()
        self.mlp = MLP(d_embd=self.d_embd)
        self.ln2 = nn.LayerNorm()

    def __call__(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


def main():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1, 10, 512))

    #
    # net = Block(n_heads=8, d_embd=512)
    #
    #
    # params = net.init(key, x)
    # print(jax.tree_map(lambda x: x.shape, params))
    # print(net.apply(params, x).shape)
    #
    # params = jax.vmap(net.init)(jnp.stack([key for _ in range(3)]), jnp.stack([x for _ in range(3)]))
    # print(jax.tree_map(lambda x: x.shape, params))

    net = MultiHeadAttention(n_heads=8, d_embd=512)
    params = net.init(key, x)
    print(jax.tree_map(lambda x: x.shape, params))

    a, b = net.apply(params, x, None, method=net.forward_recurrent)
    print(a.shape)
    print(b)


def main():
    key = jax.random.PRNGKey(0)
    b, h, t, d = 16, 12, 100, 32
    q, k, v = jax.random.normal(key, (3, b, h, t, d))
    print(q.shape, k.shape, v.shape)

    # ------ PARALLEL FORWARD ------
    out1 = ret_parallel_forward((q, k, v))

    # ------ RECURRENT FORWARD ------
    qr = rearrange(q, 'b h t d -> t b h 1 d')
    kr = rearrange(k, 'b h t d -> t b h 1 d')
    vr = rearrange(v, 'b h t d -> t b h 1 d')
    state = jnp.zeros((b, h, d, d))
    state, out2 = jax.lax.scan(ret_recurrent_forward, state, (qr, kr, vr))
    out2 = rearrange(out2, 't b h 1 d -> b h t d')

    print(f'{q.shape=} {k.shape=} {v.shape=}')
    print(f'{out1.shape=} {out2.shape=}')
    print(jnp.allclose(out1, out2, atol=1e-4))


def main():
    key = jax.random.PRNGKey(0)
    net = MultiHeadAttention(n_heads=8, d_embd=512)

    params = net.init(key, jnp.zeros((1, 1, 512)))

    b, t, d = 16, 100, 512
    x = jax.random.normal(key, (b, t, d))

    # ------ PARALLEL FORWARD ------
    out1 = net.apply(params, x, method=net.forward_parallel)
    print(out1.shape)

    # ------ RECURRENT FORWARD ------
    xr = rearrange(x, 'b t d -> t b 1 d')
    state = jnp.zeros((b, 8, 64, 64))

    from functools import partial
    forward_recurrent = partial(net.apply, params, method=net.forward_recurrent)
    state, out2 = jax.lax.scan(forward_recurrent, state, xr)
    print(state.shape, out2.shape)
    out2 = rearrange(out2, 't b 1 d -> b t d')

    print(f'{out1.shape=} {out2.shape=}')
    print(jnp.allclose(out1, out2, atol=1e-3))

    print(out1[0, 0, :10])
    print('-----')
    print(out2[0, 0, :10])


if __name__ == '__main__':
    main()
