import jax
import jax.numpy as jnp
from flax import linen as nn
from einops import rearrange


def ret_parallel_forward(qkv):
    q, k, v = qkv  # T, D
    T, D = q.shape
    mask = jnp.tril(jnp.ones((T, T), dtype=bool))  # causal masking
    attn = mask * (q @ k.T)
    out = attn @ v
    return out


def ret_recurrent_forward(state, qkv):
    # state: D, D
    q, k, v = qkv  # T, D with T=1
    T, D = q.shape
    assert T == 1
    state = state + k.T * v  # D, D
    out = q @ state  # 1, D
    return state, out  # state: D, D; out: 1, D


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
        T, D = x.shape
        assert D == self.d_embd
        kqv = self.dense1(x)  # (T, 3 * D)
        k, q, v = jnp.split(kqv, 3, axis=-1)  # (T, D)
        k = rearrange(k, 't (h d) -> h t d', h=self.n_heads)
        q = rearrange(q, 't (h d) -> h t d', h=self.n_heads)
        v = rearrange(v, 't (h d) -> h t d', h=self.n_heads)
        x = jax.vmap(ret_parallel_forward)((q, k, v))  # (H, T, Dh)
        x = rearrange(x, 'h t d -> t (h d)')  # (T, D)
        x = self.dense2(x)  # (T, D)
        return x

    def forward_recurrent(self, state, x):
        T, D = x.shape
        assert D == self.d_embd and T == 1
        kqv = self.dense1(x)  # (T, 3 * D)
        k, q, v = jnp.split(kqv, 3, axis=-1)  # (T, D)
        k = rearrange(k, 't (h d) -> h t d', h=self.n_heads)
        q = rearrange(q, 't (h d) -> h t d', h=self.n_heads)
        v = rearrange(v, 't (h d) -> h t d', h=self.n_heads)
        state, x = jax.vmap(ret_recurrent_forward)(state, (q, k, v))  # (H, Dh, Dh), (H, T, Dh)
        x = rearrange(x, 'h t d -> t (h d)')  # (T, D)
        x = self.dense2(x)  # (T, D)
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
        T, D = x.shape
        assert D == self.d_embd
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_recurrent(self, state, x):
        T, D = x.shape
        assert D == self.d_embd and T == 1
        state, xt = self.mha.forward_recurrent(state, self.ln1(x))
        x = x + xt
        x = x + self.mlp(self.ln2(x))
        return state, x


class Transformer(nn.Module):
    n_actions: int
    n_steps: int
    n_layers: int
    n_heads: int
    d_embd: int

    def setup(self):
        self.embed_obs = nn.Dense(features=self.d_embd)
        self.embed_action = nn.Embed(self.n_actions, features=self.d_embd)
        self.embed_reward = nn.Dense(features=self.d_embd)
        self.embed_time = nn.Embed(self.n_steps, features=self.d_embd)

        self.blocks = [Block(n_heads=self.n_heads, d_embd=self.d_embd) for _ in range(self.n_layers)]
        self.ln = nn.LayerNorm()

        self.actor = nn.Dense(features=self.n_actions)
        self.critic = nn.Dense(features=1)

    def __call__(self, state, action, reward, time):
        return self.forward_parallel(state, action, reward, time)

    def forward_parallel(self, obs, action, reward, time):
        obs = self.embed_obs(obs)  # T, D
        action = self.embed_action(action)  # T, D
        reward = self.embed_reward(reward)  # T, D
        time = self.embed_time(time)  # T, D
        x = obs + action + reward + time
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        logits = self.actor(x)  # T, A
        values = self.critic(x)  # T, 1
        return logits, values

    def forward_recurrent(self, state, oart):
        obs, action, reward, time = oart
        obs = self.embed_obs(obs)  # T, D
        action = self.embed_action(action)  # T, D
        reward = self.embed_reward(reward)  # T, D
        time = self.embed_time(time)  # T, D
        x = obs + action + reward + time
        state_out = [None] * self.n_layers
        for i in range(self.n_layers):
            state_out[i], x = self.blocks[i].forward_recurrent(state[i], x)
        x = self.ln(x)
        logits = self.actor(x)  # T, A
        values = self.critic(x)  # T, 1
        return state_out, (logits, values)

    def get_init_state(self, bs):
        self.d_head = self.d_embd // self.n_heads
        return [jnp.zeros((bs, self.n_heads, self.d_head, self.d_head)) for _ in range(self.n_layers)]


def main():
    rng = jax.random.PRNGKey(0)
    net = Transformer(n_actions=10, n_steps=128, n_layers=3, n_heads=4, d_embd=128)
    bs, t, d = 8, 32, 128

    # ------ INIT ------
    obs = jnp.zeros((t, 64))
    action = jnp.zeros((t, ), dtype=jnp.int32)
    reward = jnp.zeros((t, 1))
    time = jnp.zeros((t, ), dtype=jnp.int32)
    rng, _rng = jax.random.split(rng)
    params = net.init(rng, obs, action, reward, time)
    print(jax.tree_map(lambda x: x.shape, params))

    rng, *_rng = jax.random.split(rng, 1+4)
    obs = jax.random.normal(_rng[0], (bs, t, 64))
    action = jax.random.randint(_rng[1], (bs, t, ), 0, 10)
    reward = jax.random.normal(_rng[2], (bs, t, 1))
    time = jax.random.randint(_rng[3], (bs, t, ), 0, 10)

    from functools import partial
    # ------ PARALLEL FORWARD ------
    forward_parallel = partial(net.apply, params)
    forward_parallel = jax.vmap(forward_parallel, in_axes=(0, 0, 0, 0))

    logits1, values1 = forward_parallel(obs, action, reward, time)
    print('logits, values shapes')
    print(logits1.shape, values1.shape)

    print(logits1.mean(), logits1[0, 0, :10])

    # ------ RECURRENT FORWARD ------
    forward_recurrent = partial(net.apply, params, method=net.forward_recurrent)
    forward_recurrent = jax.vmap(forward_recurrent, in_axes=(0, (0, 0, 0, 0)))

    obs_r = rearrange(obs, 'b t d -> t b 1 d')
    action_r = rearrange(action, 'b t -> t b 1')
    reward_r = rearrange(reward, 'b t d -> t b 1 d')
    time_r = rearrange(time, 'b t -> t b 1')

    state = net.get_init_state(bs)
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
