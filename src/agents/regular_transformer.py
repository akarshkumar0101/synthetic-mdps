import jax
import jax.numpy as jnp
from flax import linen as nn


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        d_embd = x.shape[-1]
        x = nn.Dense(features=4 * d_embd)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=d_embd)(x)
        return x


class Block(nn.Module):
    n_heads: int
    mask_type: str

    def setup(self):
        self.ln1 = nn.LayerNorm()
        self.mha = nn.MultiHeadDotProductAttention(num_heads=self.n_heads)
        self.ln2 = nn.LayerNorm()
        self.mlp = MLP()

    def __call__(self, x):
        if self.mask_type == "causal":
            mask = jnp.tril(jnp.ones((x.shape[0], x.shape[0]), dtype=bool))
        elif self.mask_type == "eye":
            mask = jnp.eye(x.shape[0], dtype=bool)
        else:
            raise NotImplementedError
        x = x + self.mha(self.ln1(x), mask=mask, sow_weights=True)
        x = x + self.mlp(self.ln2(x))
        return x

    # def __call__(self, kv, q=None):
    #     if q is None:
    #         q = kv

    #     if self.mask_type == "causal":
    #         print("Using causal mask")
    #         mask = jnp.tril(jnp.ones((q.shape[0], kv.shape[0]), dtype=bool))
    #     elif self.mask_type == "eye":
    #         print("Using eye mask")
    #         mask = jnp.eye(q.shape[0], dtype=bool)
    #     else:
    #         raise NotImplementedError

    #     x = q + self.mha(self.ln1(q), self.ln1(kv), mask=mask, sow_weights=True)
    #     x = x + self.mlp(self.ln2(x))

    #     # x = x + self.mha(self.ln1(x), mask=mask, sow_weights=True)
    #     # x = x + self.mlp(self.ln2(x))
    #     return x


# class Transformer(nn.Module):
#     n_acts: int
#     n_layers: int
#     n_heads: int
#     d_embd: int
#     n_steps: int
#
#     def setup(self):
#         self.embed_obs = nn.Dense(features=self.d_embd)
#         self.embed_act = nn.Embed(num_embeddings=self.n_acts, features=self.d_embd)
#         self.embed_time = nn.Embed(num_embeddings=self.n_steps, features=self.d_embd)
#
#         self.blocks = [Block(n_heads=self.n_heads) for _ in range(self.n_layers)]
#         self.ln = nn.LayerNorm()
#         self.actor = nn.Dense(features=self.n_acts, kernel_init=nn.initializers.orthogonal(0.01))  # T, A
#         self.critic = nn.Dense(features=1)  # T, 1
#
#     def __call__(self, obs, act):  # obs: (T, O), # act: (T, )
#         act = jnp.concatenate([jnp.zeros_like(act[:1]), act[:-1]])
#
#         x_obs = self.embed_obs(obs)  # (T, D)
#         x_act = self.embed_act(act)  # (T, D)
#         x_time = self.embed_time(jnp.arange(self.n_steps))  # (T, D)
#         x = x_obs + x_act + x_time
#
#         for block in self.blocks:
#             x = block(x)
#         x = self.ln(x)
#         logits, val = self.actor(x), self.critic(x)  # (T, A) and (T, 1)
#         return logits, val[..., 0]


class BCTransformer(nn.Module):
    d_obs: int
    d_act: int
    n_layers: int
    n_heads: int
    d_embd: int
    ctx_len: int

    mask_type: str = "causal"

    def setup(self):
        self.embed_obs = nn.Dense(features=self.d_embd)
        self.embed_act = nn.Dense(features=self.d_embd)
        self.embed_pos = nn.Embed(num_embeddings=self.ctx_len, features=self.d_embd)

        self.blocks = [Block(n_heads=self.n_heads, mask_type=self.mask_type) for _ in range(self.n_layers)]
        self.ln = nn.LayerNorm()

        self.predict_obs = nn.Dense(features=self.d_obs, kernel_init=nn.initializers.normal(1e-4))
        self.predict_act = nn.Dense(features=self.d_act, kernel_init=nn.initializers.normal(1e-4))

    def __call__(self, obs, act):  # obs: (T, Do), act: (T, Da)
        assert obs.shape[0] == act.shape[0]
        T, Do = obs.shape
        assert T <= self.ctx_len

        x_obs = self.embed_obs(obs)  # (T, D)
        x_act = self.embed_act(act)  # (T, D)
        x_pos = self.embed_pos(jnp.arange(T))  # (T, D)

        x = x_pos + x_obs + x_act
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)

        obs_pred = self.predict_obs(x)
        act_pred = self.predict_act(x)
        return dict(obs=obs_pred, act=act_pred)


class TrajectoryTransformer(nn.Module):
    d_obs: int
    d_act: int
    n_layers: int
    n_heads: int
    d_embd: int
    ctx_len: int

    mask_type: str = "causal"

    def setup(self):
        self.embed_done = nn.Dense(features=self.d_embd)
        self.embed_rtg = nn.Dense(features=self.d_embd)
        self.embed_obs = nn.Dense(features=self.d_embd)
        self.embed_act = nn.Dense(features=self.d_embd)
        self.embed_rew = nn.Dense(features=self.d_embd)

        self.embed_pos = nn.Embed(num_embeddings=self.ctx_len, features=self.d_embd)

        self.blocks = [Block(n_heads=self.n_heads, mask_type=self.mask_type) for _ in range(self.n_layers)]
        self.ln = nn.LayerNorm()

        self.predict_done = nn.Dense(features=1, kernel_init=nn.initializers.normal(1e-4))  # T, Da
        self.predict_rtg = nn.Dense(features=1, kernel_init=nn.initializers.normal(1e-4))  # T, Da
        self.predict_obs = nn.Dense(features=self.d_obs, kernel_init=nn.initializers.normal(1e-4))  # T, Da
        self.predict_act = nn.Dense(features=self.d_act, kernel_init=nn.initializers.normal(1e-4))  # T, Da
        self.predict_rew = nn.Dense(features=1, kernel_init=nn.initializers.normal(1e-4))  # T, Da

    def __call__(self, done, rtg, obs, act, rew):  # done: (T, ), rtg: (T, ), obs: (T, Do),  act: (T, Da), rew: (T, )
        assert obs.shape[0] <= self.ctx_len
        T, _ = obs.shape

        x = self.embed_pos(jnp.arange(T))  # (T, D)
        x = x + self.embed_obs(obs)

        if done is not None:
            x_done = self.embed_done(done.astype(jnp.float32)[..., None])
            x = x + x_done
        if rtg is not None:
            x_rtg = self.embed_rtg(rtg[..., None])
            x = x + x_rtg
        if act is not None:
            x_act = self.embed_act(act)
            x_act_prv = jnp.concatenate([jnp.zeros((1, self.d_embd)), x_act[:-1]], axis=0)
            x = x + x_act_prv
        if rew is not None:
            x_rew = self.embed_rew(rew[..., None])
            x_rew_prv = jnp.concatenate([jnp.zeros((1, self.d_embd)), x_rew[:-1]], axis=0)
            x = x + x_rew_prv

        for block in self.blocks:
            x = block(x)
        x = self.ln(x)

        done_nxt_pred = self.predict_done(x)
        rtg_nxt_pred = self.predict_rtg(x)
        obs_nxt_pred = self.predict_obs(x)
        act_pred = self.predict_act(x)
        rew_pred = self.predict_rew(x)
        return dict(done_nxt_pred=done_nxt_pred, rtg_nxt_pred=rtg_nxt_pred, obs_nxt_pred=obs_nxt_pred, act_pred=act_pred, rew_pred=rew_pred)

