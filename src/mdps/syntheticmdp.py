import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces
import flax.linen as nn


class SyntheticMDP(environment.Environment):
    def __init__(self, n_acts,
                 model_init, model_trans, model_obs, model_rew, iid_multi_trans=True):
        super().__init__()
        self.n_acts = n_acts
        self.iid_multi_trans = iid_multi_trans
        assert self.iid_multi_trans  # make sure other case is implemented properly

        self.model_init = model_init  # rng to state
        self.model_trans = model_trans  # (state, rng to state) if iid_multi_trans else (state, action, rng to state)
        self.model_obs = model_obs  # state to obs
        self.model_rew = model_rew  # state to R

    def sample_params(self, rng):
        _rng_init, _rng_trans, _rng_obs, _rng_rew = jax.random.split(rng, 4)

        params_init = self.model_init.init(_rng_init, rng)
        state = self.model_init.apply(params_init, rng)
        if self.iid_multi_trans:
            _rng_trans = jnp.stack(jax.random.split(_rng_trans, self.n_acts))
            params_trans = jax.vmap(self.model_trans.init, in_axes=(0, None, None))(_rng_trans, state, rng)
        else:
            action = jnp.zeros((), dtype=jnp.int32)
            params_trans = self.model_trans.init(_rng_trans, state, action, rng)
        params_obs = self.model_obs.init(_rng_obs, state)
        params_rew = self.model_rew.init(_rng_rew, state)
        return dict(params_init=params_init, params_trans=params_trans,
                    params_obs=params_obs, params_rew=params_rew)

    def reset_env(self, rng, params):
        """Performs resetting of environment."""
        params_init, params_trans = params['params_init'], params['params_trans']
        params_obs, params_rew = params['params_obs'], params['params_rew']

        state = self.model_init.apply(params_init, rng)
        obs = self.model_obs.apply(params_obs, state)
        return obs, state

    def step_env(self, rng, state, action, params):
        """Performs step transitions in the environment."""
        params_init, params_trans = params['params_init'], params['params_trans']
        params_obs, params_rew = params['params_obs'], params['params_rew']

        if self.iid_multi_trans:
            state = jax.vmap(self.model_trans.apply, in_axes=(0, None, None))(params_trans, state, rng)[action]
        else:
            state = self.model_trans.apply(params_trans, state, action)
        obs = self.model_obs.apply(params_obs, state)
        rew = self.model_rew.apply(params_rew, state)
        done = jnp.zeros((), dtype=jnp.bool_)
        info = {}
        return obs, state, rew, done, info

    @property
    def name(self) -> str:
        return "SyntheticMDP"

    @property
    def num_actions(self) -> int:
        return self.n_acts

    def action_space(self, params):
        return spaces.Discrete(self.n_acts)

    def observation_space(self, params):
        params_obs = params['params_obs']
        return self.model_obs.observation_space(params_obs)

# class MultiActionFunction(nn.Module):
#     model_class: nn.Module
#     n_acts: int
#
#     def setup(self, ):
#         batch_model_class = nn.vmap(self.model_class, in_axes=0, out_axes=0,
#                                     variable_axes={'params': 0}, split_rngs={'params': True})
#         self.model = batch_model_class()
#
#     def __call__(self, state, action, rng):
#         state = repeat(state, '... -> a ...', a=self.n_acts)
#         state = self.model(state)[action]
#         return state / jnp.linalg.norm(state)
#
# # change to function which takes apply_fn, params_stack, etc.
#
# def multifn(apply_fn, params, state, action):
#     return jax.vmap(apply_fn, in_axes=(0, None))(params, state)[action]


def main():
    rng = jax.random.PRNGKey(0)
    d_state = 10
    d_obs = 32
    n_acts = 4

    from functools import partial

    #
    # state = jnp.zeros((d_state,))
    # action = jnp.zeros((), dtype=jnp.int32)
    # params = model_trans.init(rng, state, action)
    # print(jax.tree_map(lambda x: x.shape, params))

    # Dense = partial(nn.Dense, features=d_state, use_bias=False)
    # BatchDense = nn.vmap(Dense, in_axes=0, out_axes=0, variable_axes={'params': 0}, split_rngs={'params': True})
    # model = BatchDense()
    # # model = nn.Dense(features=d_state, use_bias=False)
    #
    # rng, _rng = jax.random.split(rng)
    # state = jnp.zeros((4, d_state, ))
    # params = model.init(_rng, state)
    # print(jax.tree_map(lambda x: x.shape, params))

    # Dense = partial(nn.Dense, features=d_state, use_bias=False)
    # model_trans = MultiActionFunction(Dense, n_acts)
    # state = jnp.zeros((d_state,))
    # action = jnp.zeros((), dtype=jnp.int32)
    # params = model_trans.init(rng, state, action)
    # print(jax.tree_map(lambda x: x.shape, params))

    model_trans = nn.Sequential([nn.Dense(features=d_state, use_bias=False), lambda x: x / jnp.linalg.norm(x)])
    model_obs = nn.Dense(features=d_obs, use_bias=False)
    model_rew = nn.Dense(features=1, use_bias=False)

    mdp = SyntheticMDP((d_state, ), (d_obs, ), n_acts, model_trans, model_obs, model_rew)

    rng, _rng = jax.random.split(rng)
    env_params = mdp.sample_params(_rng)
    print(jax.tree_map(lambda x: x.shape, env_params))

    rng, _rng = jax.random.split(rng)
    obs, env_state = mdp.reset_env(_rng, env_params)

    for i in range(100):
        rng, _rng = jax.random.split(rng)
        action = mdp.action_space(env_params).sample(_rng)
        action = jnp.zeros_like(action)
        # print('env_state: ')
        # print(jax.tree_map(lambda x: x.shape, env_state))
        # print('action: ')
        # print(jax.tree_map(lambda x: x.shape, action))
        rng, _rng = jax.random.split(rng)
        obs, env_state, rew, done, info = mdp.step(_rng, env_state, action, env_params)

        print(env_state[:4])


if __name__ == '__main__':
    main()
