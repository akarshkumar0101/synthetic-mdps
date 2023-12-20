import flax.linen as nn
import jax
from gymnax.environments import environment


class SyntheticMDP(environment.Environment):
    def __init__(self, model_init, model_trans, model_obs, model_rew, model_done):
        super().__init__()
        self.model_init = model_init  # rng -> state
        self.model_trans = model_trans  # state, action, rng -> state
        self.model_obs = model_obs  # state -> obs
        self.model_rew = model_rew  # state -> R
        self.model_done = model_done  # state -> done

    def sample_params(self, rng):
        _rng_init, _rng_trans, _rng_obs, _rng_rew, _rng_done = jax.random.split(rng, 5)
        params_init = self.model_init.init(_rng_init, rng)

        state = self.model_init.apply(params_init, rng)
        action = self.model_trans.action_space(None).sample(rng)

        params_trans = self.model_trans.init(_rng_trans, rng, state, action)
        params_obs = self.model_obs.init(_rng_obs, state)
        params_rew = self.model_rew.init(_rng_rew, state)
        params_done = self.model_done.init(_rng_done, state)
        return dict(params_init=params_init, params_trans=params_trans,
                    params_obs=params_obs, params_rew=params_rew, params_done=params_done)

    def reset_env(self, rng, params):
        """Performs resetting of environment."""
        ks = ["params_init", "params_trans", "params_obs", "params_rew", "params_done"]
        params_init, params_trans, params_obs, params_rew, params_done = [params[k] for k in ks]

        state = self.model_init.apply(params_init, rng)
        obs = self.model_obs.apply(params_obs, state)
        return obs, state

    def step_env(self, rng, state, action, params):
        """Performs step transitions in the environment."""
        ks = ["params_init", "params_trans", "params_obs", "params_rew", "params_done"]
        params_init, params_trans, params_obs, params_rew, params_done = [params[k] for k in ks]

        state = self.model_trans.apply(params_trans, rng, state, action)
        obs = self.model_obs.apply(params_obs, state)
        rew = self.model_rew.apply(params_rew, state)
        done = self.model_done.apply(params_done, state)
        info = {}
        return obs, state, rew, done, info

    @property
    def name(self) -> str:
        return "SyntheticMDP"

    def action_space(self, params):
        # return self.model_trans.action_space(params['params_trans']) #TODO look into this...
        return self.model_trans.action_space(None)

    def observation_space(self, params):
        return self.model_obs.observation_space(params['params_obs'])


class IdentityObs(nn.Module):
    def setup(self):
        pass

    def __call__(self, state):
        return state


class NeverDone(nn.Module):
    def setup(self):
        pass

    def __call__(self, state):
        return False
