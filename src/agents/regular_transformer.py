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
        self.mha = nn.MultiHeadDotProductAttention(num_heads=self.n_heads)
        self.ln1 = nn.LayerNorm()
        self.mlp = MLP()
        self.ln2 = nn.LayerNorm()

    def __call__(self, kv, q=None):
        if q is None:
            q = kv

        if self.mask_type == "causal":
            print("Using causal mask")
            mask = jnp.tril(jnp.ones((q.shape[0], kv.shape[0]), dtype=bool))
        elif self.mask_type == "eye":
            print("Using eye mask")
            mask = jnp.eye(q.shape[0], dtype=bool)
        else:
            raise NotImplementedError

        x = q + self.mha(self.ln1(q), self.ln1(kv), mask=mask, sow_weights=True)
        x = x + self.mlp(self.ln2(x))
        return x


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
    max_dt: int = 16

    def setup(self):
        self.embed_obs = nn.Dense(features=self.d_embd)
        self.embed_act = nn.Dense(features=self.d_embd)
        self.embed_pos = nn.Embed(num_embeddings=self.ctx_len, features=self.d_embd)
        self.embed_dt = nn.Embed(num_embeddings=self.max_dt, features=self.d_embd)

        self.blocks = [Block(n_heads=self.n_heads, mask_type=self.mask_type) for _ in range(self.n_layers)]
        self.ln = nn.LayerNorm()

        self.actor = nn.Dense(features=self.d_act, kernel_init=nn.initializers.orthogonal(0.01))  # T, Da
        self.wm = nn.Dense(features=self.d_obs, kernel_init=nn.initializers.orthogonal(0.01))  # T, Do

    def __call__(self, obs, act, time):  # obs: (T, Do), # act: (T, Da), # time: (T, )
        assert obs.shape[0] == act.shape[0]
        assert obs.shape[0] <= self.ctx_len
        T, _ = obs.shape
        time_prv, time_now, time_nxt = time[:-2], time[1:-1], time[2:]
        dt = time_nxt - time_now

        x_obs = self.embed_obs(obs)  # (T, D)
        x_act = self.embed_act(act)  # (T, D)
        x_pos = self.embed_pos(jnp.arange(T))  # (T, D)
        x_dt = self.embed_dt(dt)

        obs_prv, obs_now, obs_nxt = obs[:-2], obs[1:-1], obs[2:]
        act_prv, act_now, act_nxt = act[:-2], act[1:-1], act[2:]
        x_obs_prv, x_obs_now, x_obs_nxt = x_obs[:-2], x_obs[1:-1], x_obs[2:]
        x_act_prv, x_act_now, x_act_nxt = x_act[:-2], x_act[1:-1], x_act[2:]
        x_pos_prv, x_pos_now, x_pos_nxt = x_pos[:-2], x_pos[1:-1], x_pos[2:]

        x = x_pos_now + x_obs_now + x_act_prv + x_dt
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)

        act_now_pred = self.actor(x)
        obs_nxt_pred = self.wm(x)

        result = dict(obs_prv=obs_prv, obs_now=obs_now, obs_nxt=obs_nxt,
                      act_prv=act_prv, act_now=act_now, act_nxt=act_nxt,
                      act_now_pred=act_now_pred, obs_nxt_pred=obs_nxt_pred)
        return result


def calc_entropy_stable(logits, axis=-1):
    logits = jax.nn.log_softmax(logits, axis=axis)
    probs = jnp.exp(logits)
    logits = jnp.where(probs == 0, 0., logits)  # replace -inf with 0
    return -(probs * logits).sum(axis=axis)
