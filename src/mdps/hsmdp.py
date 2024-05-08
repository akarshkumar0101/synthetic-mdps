







class Init:
    def __init__(self, n, d_state, temperature=1., std=0., constraint='clip'):
        self.n, self.temperature = n, temperature
        self.d_state, self.std = d_state, std
        self.constraint = constraint

    def sample_params(self, rng):
        logits = jax.random.normal(rng, (self.n,))
        mean = jax.random.normal(rng, (self.d_state,))
        return dict(logits=logits, mean=mean)

    def __call__(self, rng, params):
        state = params['mean'] + self.std * jax.random.normal(rng, (self.d_state,))
        state = constrain_state(state, self.constraint)
        state2 = jax.random.categorical(rng, params['logits'] / self.temperature, axis=-1)
        return (state, state2)


