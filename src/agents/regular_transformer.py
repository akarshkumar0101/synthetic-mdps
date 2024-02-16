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

    def setup(self):
        self.mha = nn.MultiHeadDotProductAttention(num_heads=self.n_heads)
        self.ln1 = nn.LayerNorm()
        self.mlp = MLP()
        self.ln2 = nn.LayerNorm()

    def __call__(self, x):
        mask = jnp.tril(jnp.ones((x.shape[0], x.shape[0]), dtype=bool))
        # x = x + self.mha(self.ln1(x))
        temp = self.ln1(x)
        x = x + self.mha(temp, temp, mask=mask)  # TODO: use new version of jax so we don't need to do this

        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    n_acts: int
    n_layers: int
    n_heads: int
    d_embd: int
    n_steps: int

    def setup(self):
        self.embed_obs = nn.Dense(features=self.d_embd)
        self.embed_act = nn.Embed(num_embeddings=self.n_acts, features=self.d_embd)
        self.embed_time = nn.Embed(num_embeddings=self.n_steps, features=self.d_embd)

        self.blocks = [Block(n_heads=self.n_heads) for _ in range(self.n_layers)]
        self.ln = nn.LayerNorm()
        self.actor = nn.Dense(features=self.n_acts, kernel_init=nn.initializers.orthogonal(0.01))  # T, A
        self.critic = nn.Dense(features=1)  # T, 1

    def __call__(self, obs, act):  # obs: (T, O), # act: (T, )
        act = jnp.concatenate([jnp.zeros_like(act[:1]), act[:-1]])

        x_obs = self.embed_obs(obs)  # (T, D)
        x_act = self.embed_act(act)  # (T, D)
        x_time = self.embed_time(jnp.arange(self.n_steps))  # (T, D)
        x = x_obs + x_act + x_time

        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        logits, val = self.actor(x), self.critic(x)  # (T, A) and (T, 1)
        return logits, val[..., 0]


class BCTransformer(nn.Module):
    n_acts: int
    n_layers: int
    n_heads: int
    d_embd: int
    n_steps: int

    def setup(self):
        self.embed_obs = nn.Dense(features=self.d_embd)
        self.embed_act = nn.Embed(num_embeddings=self.n_acts, features=self.d_embd)
        self.embed_time = nn.Embed(num_embeddings=self.n_steps, features=self.d_embd)

        self.blocks = [Block(n_heads=self.n_heads) for _ in range(self.n_layers)]
        self.ln = nn.LayerNorm()
        self.actor = nn.Dense(features=self.n_acts, kernel_init=nn.initializers.orthogonal(0.01))  # T, A

    def __call__(self, obs, act):  # obs: (T, O), # act: (T, )
        assert obs.shape[0] == act.shape[0]
        assert obs.shape[0] <= self.n_steps
        T, _ = obs.shape

        act_p = jnp.concatenate([jnp.zeros_like(act[:1]), act[:-1]], axis=0)

        x_obs = self.embed_obs(obs)  # (T, D)
        x_act = self.embed_act(act_p)  # (T, D)
        x_time = self.embed_time(jnp.arange(T))  # (T, D)
        x = x_obs + x_act + x_time

        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        logits = self.actor(x)  # (T, A)
        return logits


class WMTransformer(nn.Module):
    n_acts: int
    n_layers: int
    n_heads: int
    d_embd: int
    n_steps: int
    d_obs: int

    def setup(self):
        self.embed_obs = nn.Dense(features=self.d_embd)
        self.embed_act = nn.Embed(num_embeddings=self.n_acts, features=self.d_embd)
        self.embed_time = nn.Embed(num_embeddings=self.n_steps, features=self.d_embd)

        self.blocks = [Block(n_heads=self.n_heads) for _ in range(self.n_layers)]
        self.ln = nn.LayerNorm()
        self.lin_out = nn.Dense(features=self.d_obs, kernel_init=nn.initializers.orthogonal(0.01))  # T, A

    def __call__(self, obs, act):  # obs: (T, O), # act: (T, )
        x_obs = self.embed_obs(obs)  # (T, D)
        x_act = self.embed_act(act)  # (T, D)
        x_time = self.embed_time(jnp.arange(self.n_steps))  # (T, D)
        x = x_obs + x_act + x_time

        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        obs_pred = self.lin_out(x)  # (T, A)
        return obs_pred


def calc_entropy_stable(logits, axis=-1):
    logits = jax.nn.log_softmax(logits, axis=axis)
    probs = jnp.exp(logits)
    logits = jnp.where(probs == 0, 0., logits)  # replace -inf with 0
    return -(probs * logits).sum(axis=axis)
