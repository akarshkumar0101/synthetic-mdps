import torch
from torch import nn
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import numpy as np
import flax
from flax import linen as nn


def create_model():
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.Sigmoid(),
        nn.Linear(8, 8),
        nn.Sigmoid(),
        nn.Linear(8, 8),
        nn.Softmax(dim=-1),
    )


def main():
    m0 = create_model()
    m1 = create_model()
    m2 = create_model()
    x = torch.randn(3, 4)

    print(m0(x[0]))
    print(m1(x[1]))
    print(m2(x[2]))

    d0 = dict(m0.named_parameters())
    d1 = dict(m1.named_parameters())
    d2 = dict(m2.named_parameters())
    d = {}
    for k, v in d0.items():
        d[k] = torch.stack([v, d1[k], d2[k]], dim=0)

    call = torch.func.functional_call
    call = torch.vmap(call, in_dims=(None, 0, 0))
    print(call(m0, d, x))


# def func(W, x):
#     print(W.shape, x.shape)
#     print('hello')
#     return W @ x

# def main():
#     func_vmap = jax.vmap(func)
#
#     print(func_vmap(np.random.randn(3, 200, 100), np.random.randn(3, 100)).shape)

class CNN(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    return x


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Line


def main():
    pass


import equinox as eqx
import jax

class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_size, out_size, key):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (out_size, in_size))
        self.bias = jax.random.normal(bkey, (out_size,))

    def __call__(self, x):
        return self.weight @ x + self.bias


if __name__ == '__main__':
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (10,))
    net = Linear(10, 20, rng)
    y = net(x)
    print(jax.tree_map(lambda x: x.shape, net))
    print(x.shape, y.shape)

    x = jax.random.normal(rng, (10,))
    rng, *_rng = jax.random.split(rng, 1+4)
    net = jax.vmap(Linear, in_axes=(None, None, 0))(10, 20, jnp.stack(_rng))
    y = net(x)
    print(jax.tree_map(lambda x: x.shape, net))
    # print(x.shape, y.shape)

