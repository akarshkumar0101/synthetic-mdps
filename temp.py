import jax
import jax.numpy as jnp
from jax.random import split


def run_it(rng):
    a = jax.random.normal(rng, (1000, 1000))
    b = jax.random.normal(rng, (1000, 1000))
    c = a@b
    return c.sum()


def main():
    train = jax.pmap(run_it, )
    rng = jax.random.PRNGKey(0)

    cs = train(split(rng, 6))
    print(cs)


if __name__ == '__main__':
    main()

