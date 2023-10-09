import jax
import jax.numpy as jnp
from flax import linen as nn

from einops import rearrange, repeat

import math


class MultiHeadAttention(nn.Module):
    n_heads: int
    d_embd: int

    def setup(self):
        assert self.d_embd % self.n_heads == 0
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

    def forward_recurrent(self, x, state=None):
        return x, state


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

    net = Block(n_heads=8, d_embd=512)

    x = jax.random.normal(key, (1, 10, 512))

    params = net.init(key, x)
    print(jax.tree_map(lambda x: x.shape, params))
    print(net.apply(params, x).shape)

    params = jax.vmap(net.init)(jnp.stack([key for _ in range(3)]), jnp.stack([x for _ in range(3)]))
    print(jax.tree_map(lambda x: x.shape, params))


if __name__ == '__main__':
    main()
