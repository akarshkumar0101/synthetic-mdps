import torch
from torch import nn
from einops import rearrange, repeat
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_embd, n_heads):
        super().__init__()
        assert d_embd % n_heads == 0
        self.d_embd, self.n_heads, self.d_head = d_embd, n_heads, d_embd // n_heads

        self.dense1 = nn.Linear(self.d_embd, 3 * self.d_embd)
        self.dense2 = nn.Linear(self.d_embd, self.d_embd)

    def forward(self, x, mask=None):
        bs, t, d_embd = x.shape
        assert d_embd == self.d_embd

        if mask is None:  # assume causal mask
            mask = torch.tril(torch.ones((t, t), dtype=bool, device=x.device))
            mask = repeat(mask, 't t2 -> b h t t2', b=bs, h=self.n_heads)

        kqv = self.dense1(x)  # (bs, t, 3 * d_embd)
        k, q, v = torch.chunk(kqv, 3, dim=-1)  # (bs, t, d_embd)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.n_heads)  # (bs, n_heads, t, d_head)
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.n_heads)  # (bs, n_heads, t, d_head)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.n_heads)  # (bs, n_heads, t, d_head)
        attn = (q @ k.mT) * (1.0 / math.sqrt(self.d_head))  # (bs, n_heads, t, t)
        attn = attn.masked_fill(~mask, -float("inf"))
        # attn = torch.softmax(attn, dim=-1)  # (bs, n_heads, t, t)
        x = attn @ v  # (bs, n_heads, t, d_head)
        x = rearrange(x, 'b h t d -> b t (h d)')  # (bs, t, d_embd)
        x = self.dense2(x)  # (bs, t, d_embd)
        return x


class MLP(nn.Module):
    def __init__(self, d_embd):
        super().__init__()
        self.dense1 = nn.Linear(d_embd, 4 * d_embd)
        self.dense2 = nn.Linear(4 * d_embd, d_embd)

    def forward(self, x):
        x = self.dense1(x)
        x = nn.functional.gelu(x)
        x = self.dense2(x)
        return x


class Block(nn.Module):

    def __init__(self, d_embd, n_heads):
        super().__init__()
        self.mha = MultiHeadAttention(d_embd=d_embd, n_heads=n_heads)
        self.ln1 = nn.LayerNorm((d_embd, ))
        self.mlp = MLP(d_embd=d_embd)
        self.ln2 = nn.LayerNorm((d_embd, ))

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


def main():
    torch.manual_seed(0)

    net = Block(d_embd=512, n_heads=8)
    x = torch.randn(1, 40, 512)
    y = net(x)
    print(y.shape)


if __name__ == '__main__':
    main()
