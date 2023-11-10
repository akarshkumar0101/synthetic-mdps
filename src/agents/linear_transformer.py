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
    q, k, v = qkv  # D
    assert q.ndim == 1 and k.ndim == 1 and v.ndim == 1
    q, k, v = q[None, :], k[None, :], v[None, :]  # 1, D
    state = state + k.T * v  # D, D
    out = q @ state  # 1, D
    return state, out[0]  # state: D, D; out: D


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
        D, = x.shape
        assert D == self.d_embd
        kqv = self.dense1(x)  # (3 * D)
        k, q, v = jnp.split(kqv, 3, axis=-1)  # (D)
        k = rearrange(k, '(h d) -> h d', h=self.n_heads)
        q = rearrange(q, '(h d) -> h d', h=self.n_heads)
        v = rearrange(v, '(h d) -> h d', h=self.n_heads)
        state, x = jax.vmap(ret_recurrent_forward)(state, (q, k, v))  # (H, Dh, Dh), (H, Dh)
        x = rearrange(x, 'h d -> (h d)')  # (D)
        x = self.dense2(x)  # (D)
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
        D, = x.shape
        assert D == self.d_embd
        state, xt = self.mha.forward_recurrent(state, self.ln1(x))
        x = x + xt
        x = x + self.mlp(self.ln2(x))
        return state, x


class LinearTransformerAgent(nn.Module):
    n_acts: int
    n_steps: int
    n_layers: int
    n_heads: int
    d_embd: int

    def setup(self):
        self.embed_obs = nn.Dense(features=self.d_embd)
        self.embed_action = nn.Embed(self.n_acts, features=self.d_embd)
        self.embed_reward = nn.Dense(features=self.d_embd)
        self.embed_time = nn.Embed(self.n_steps, features=self.d_embd)

        self.blocks = [Block(n_heads=self.n_heads, d_embd=self.d_embd) for _ in range(self.n_layers)]
        self.ln = nn.LayerNorm()

        self.actor = nn.Dense(features=self.n_acts,
                              kernel_init=nn.initializers.orthogonal(0.01),
                              bias_init=nn.initializers.constant(0.0))
        self.critic = nn.Dense(features=1)

    def forward_parallel(self, obs, action, reward):
        assert obs.shape[0] == self.n_steps
        time = jnp.arange(self.n_steps)

        x_obs = self.embed_obs(obs)  # T, D
        x_action = self.embed_action(action)  # T, D
        x_reward = self.embed_reward(reward[..., None])  # T, D
        x_time = self.embed_time(time)  # T, D
        x = x_obs + x_action + x_reward + x_time

        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        logits = self.actor(x)  # T, A
        values = self.critic(x)  # T, 1
        return logits, values[..., 0]

    def forward_recurrent(self, state, oar):
        kTv, time = state['kTv'], state['time']
        obs, action, reward = oar
        # assert time < self.n_steps

        x_obs = self.embed_obs(obs)  # D
        x_action = self.embed_action(action)  # D
        x_reward = self.embed_reward(reward[..., None])  # D
        x_time = self.embed_time(time)  # D
        x = x_obs + x_action + x_reward + x_time

        kTv_out = [None] * self.n_layers
        for i in range(self.n_layers):
            kTv_out[i], x = self.blocks[i].forward_recurrent(kTv[i], x)
        x = self.ln(x)
        logits = self.actor(x)  # A
        values = self.critic(x)  # 1
        return dict(kTv=kTv_out, time=time+1), (logits, values[..., 0])

    def get_init_state(self, rng):
        d_head = self.d_embd // self.n_heads
        kTv = [jnp.zeros((self.n_heads, d_head, d_head)) for _ in range(self.n_layers)]
        time = jnp.zeros((), dtype=jnp.int32)
        return dict(kTv=kTv, time=time)


def main():
    from functools import partial

    rng = jax.random.PRNGKey(0)
    bs, n_steps, d = 8, 32, 128
    n_acts = 4

    net = LinearTransformerAgent(n_acts=n_acts, n_steps=n_steps, n_layers=3, n_heads=4, d_embd=128)
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
