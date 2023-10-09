import jax
import jax.numpy as jnp
import numpy as np
import flax
from flax import linen as nn

import matplotlib.pyplot as plt

from flax.training import train_state

import optax

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=128)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(features=1)(x)
        return x


def main():
    net = MLP()
    # print(net(x))
    for i in range(10, 11):
        key = jax.random.PRNGKey(i)
        x = jnp.zeros((1, 1))

        print(net.tabulate(jax.random.key(0), x))

        params = net.init(key, x)

        x = jnp.linspace(-10, 10, 1000)
        y = net.apply(params, x[:, None])[:, 0]
        # print(y.shape)

        # plt.plot(x, y)
        # plt.ylim(-1, 1)
    # plt.show()

    # tx = optax.sgd(learning_rate=1e-3, momentum=0.9)
    # state = train_state.TrainState.create(apply_fn=net.apply, params=params, tx=tx)
    #
    # def loss_fn(x):
    #     y = net.apply(state.params, x)
    #     return jnp.mean(y ** 2)
    #
    # grad_fn = jax.grad(loss_fn)
    #
    # grads = grad_fn(x[:, None])
    #
    # x = jnp.zeros((1, ))
    # key, subkey = jax.random.split(key)
    # params = net.init(subkey, x)
    # print(params['params']['Dense_0']['kernel'].shape)

    init_fn = net.init
    init_fn_vmap = jax.vmap(init_fn, in_axes=(0, None))
    key, *subkey = jax.random.split(key, num=11)
    subkey = jnp.stack(subkey, axis=0)
    # print(subkey, subkey.shape)
    x = jnp.zeros((1, ))
    params = init_fn_vmap(subkey, x)
    # print(params['params']['Dense_0']['kernel'].shape)


    def forward(params, x):
        return net.apply(params, x)

    x = jnp.zeros((1000, 1))
    params = net.init(key, x)
    print('weight shape: ', params['params']['Dense_0']['kernel'].shape)
    print(x.shape, forward(params, x).shape)

    forward_vmap = jax.vmap(forward, in_axes=(0, None))
    init_fn_vmap = jax.vmap(init_fn, in_axes=(0, None))
    key, *subkey = jax.random.split(key, num=11)
    subkey = jnp.stack(subkey, axis=0)
    params = init_fn_vmap(subkey, x)
    print('weight shape: ', params['params']['Dense_0']['kernel'].shape)
    print(x.shape, forward_vmap(params, x).shape)

    x = jnp.linspace(-10, 10, 1000)[:, None]
    print(x.shape)
    y = forward_vmap(params, x)
    print(x.shape, y.shape)

    for yi in y:
        plt.plot(x[:, 0], yi[:, 0])
    plt.show()


if __name__ == '__main__':
    main()
