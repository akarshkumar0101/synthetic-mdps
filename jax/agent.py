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

    def __call__(self, x):
        return self.forward_parallel(x)

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
        return self.forward_parallel(x)

    def forward_parallel(self, x):
        b, t, d = x.shape
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_recurrent(self, state, x):
        b, t, d = x.shape
        assert t == 1
        state, xt = self.mha.forward_recurrent(state, self.ln1(x))
        x = x + xt
        x = x + self.mlp(self.ln2(x))
        return state, x

class Transformer(nn.Module):
    n_actions: int
    n_layers: int
    n_heads: int
    d_embd: int

    def setup(self):
        self.embed_obs = nn.Dense(features=self.d_embd)
        self.embed_action = nn.Embed(self.n_actions, features=self.d_embd)
        self.embed_reward = nn.Dense(features=self.d_embd)
        self.embed_time = nn.Embed(self.n_actions, features=self.d_embd)

        self.blocks = [Block(n_heads=self.n_heads, d_embd=self.d_embd) for _ in range(self.n_layers)]
        self.ln = nn.LayerNorm()

        self.actor = nn.Dense(features=self.n_actions)
        self.critic = nn.Dense(features=1)

    def __call__(self, state, action, reward, time):
        return self.forward_parallel(state, action, reward, time)

    def forward_parallel(self, obs, action, reward, time):
        print('obs, action, reward, time shapes')
        print(obs.shape, action.shape, reward.shape, time.shape)
        obs = self.embed_obs(obs)  # B, T, D
        action = self.embed_action(action)  # B, T, D
        reward = self.embed_reward(reward)  # B, T, D
        time = self.embed_time(time)  # B, T, D
        print('obs, action, reward, time shapes')
        print(obs.shape, action.shape, reward.shape, time.shape)

        # x = obs
        # x.at[:, 1:].add(action[:, :-1])  # s_1 contains a_0, s_n contains a_{n-1}
        # x.at[:, 1:].add(reward[:, :-1])  # s_1 contains r_0, s_n contains r_{n-1}
        # x = x + time
        x = obs + action + reward + time

        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        logits = self.actor(x)  # B, T, n_actions
        values = self.critic(x)  # B, T, 1
        return logits, values

    def forward_recurrent(self, state, oart):
        obs, action, reward, time = oart
        print('obs, action, reward, time shapes')
        print(obs.shape, action.shape, reward.shape, time.shape)
        obs = self.embed_obs(obs)  # B, 1, D
        action = self.embed_action(action)  # B, 1, D
        reward = self.embed_reward(reward)  # B, 1, D
        time = self.embed_time(time)  # B, 1, D
        print('obs, action, reward, time shapes')
        print(obs.shape, action.shape, reward.shape, time.shape)

        x = obs + action + reward + time

        state_out = [None] * self.n_layers
        for i in range(self.n_layers):
            state_out[i], x = self.blocks[i].forward_recurrent(state[i], x)
        x = self.ln(x)
        logits = self.actor(x)  # B, 1, n_actions
        values = self.critic(x)  # B, 1, 1
        return state_out, (logits, values)

    def get_init_state(self, bs):
        self.d_head = self.d_embd // self.n_heads
        return [jnp.zeros((bs, self.n_heads, self.d_head, self.d_head)) for _ in range(self.n_layers)]


def main():
    rng = jax.random.PRNGKey(0)
    net = Transformer(n_actions=10, n_layers=3, n_heads=4, d_embd=128)

    b, t, d = 8, 32, 128
    obs = jnp.zeros((b, t, 64))
    action = jnp.zeros((b, t), dtype=jnp.int32)
    reward = jnp.zeros((b, t, 1))
    time = jnp.zeros((b, t), dtype=jnp.int32)

    params = net.init(rng, obs, action, reward, time)

    print(jax.tree_map(lambda x: x.shape, params))

    # ------ PARALLEL FORWARD ------
    logits1, values1 = net.apply(params, obs, action, reward, time)
    print('logits, values shapes')
    print(logits1.shape, values1.shape)

    # ------ RECURRENT FORWARD ------
    obs_r = rearrange(obs, 'b t d -> t b 1 d')
    action_r = rearrange(action, 'b t -> t b 1')
    reward_r = rearrange(reward, 'b t d -> t b 1 d')
    time_r = rearrange(time, 'b t -> t b 1')

    from functools import partial
    forward_recurrent = partial(net.apply, params, method=net.forward_recurrent)
    state = net.get_init_state(b)
    state, (logits2, values2) = jax.lax.scan(forward_recurrent, state, (obs_r, action_r, reward_r, time_r))
    logits2 = rearrange(logits2, 't b 1 d -> b t d')
    values2 = rearrange(values2, 't b 1 d -> b t d')
    print('logits, values shapes')
    print(logits2.shape, values2.shape)

    print(jnp.allclose(logits1, logits2, atol=1e-3))


if __name__ == '__main__':
    main()


"""
obs = env.reset()
action, reward = 0, 0
for i in range(1000):
    action = agent.act(obs, action, reward)
    obs, reward, done, info = env.step(action)
    
"""


def test_retnet_qkv():
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

