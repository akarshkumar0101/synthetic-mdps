from typing import Callable

import flax.linen as nn
import jax.numpy as jnp
from jax.random import split


class RandomMLP(nn.Module):
    n_layers: int
    d_hidden: int
    d_out: int
    activation: Callable

    @nn.compact
    def __call__(self, x, train=False):
        for _ in range(self.n_layers):
            x = nn.Dense(features=self.d_hidden, kernel_init=nn.initializers.normal(1),
                         bias_init=nn.initializers.normal(1))(x)
            x = nn.BatchNorm(use_running_average=not train, momentum=0., use_bias=False, use_scale=False)(x)
            x = self.activation(x)
            # x = nn.BatchNorm(use_running_average=not train, momentum=0., use_bias=False, use_scale=False)(x)
        x = nn.Dense(features=self.d_out, kernel_init=nn.initializers.normal(1), bias_init=nn.initializers.normal(1))(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0., use_bias=False, use_scale=False)(x)
        return x


def create_random_net(net, rng, x):
    """
    x should have shape (B, D)
    """
    rng, _rng = split(rng)
    params = net.init(_rng, jnp.zeros_like(x[0]))
    y, updates = net.apply(params, x, train=True, mutable=['batch_stats'])
    params['batch_stats'] = updates['batch_stats']
    y = net.apply(params, x)
    # assert jnp.allclose(y.mean(axis=0), jnp.zeros_like(y.mean(axis=0)), atol=1e-2)
    # assert jnp.allclose(y.std(axis=0), jnp.ones_like(y.std(axis=0)), atol=1e-2)
    return params


def create_random_net_normal(rng, net, batch_size, d_in):
    _rng1, _rng2 = split(rng)
    x = jax.random.normal(_rng1, (batch_size, d_in))
    params = net.init(_rng2, x[0])
    y, updates = net.apply(params, x, train=True, mutable=['batch_stats'])
    params['batch_stats'] = updates['batch_stats']
    y = net.apply(params, x)
    # assert jnp.allclose(y.mean(axis=0), jnp.zeros_like(y.mean(axis=0)), atol=1e-2)
    # assert jnp.allclose(y.std(axis=0), jnp.ones_like(y.std(axis=0)), atol=1e-2)
    return params
