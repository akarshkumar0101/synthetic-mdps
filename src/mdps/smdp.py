import jax
from gymnax.environments import environment, spaces
from jax.random import split


class SyntheticMDP(environment.Environment):
    def __init__(self, model_init, model_trans, model_obs, model_rew, model_done):
        super().__init__()
        self.model_init = model_init  # rng -> state
        self.model_trans = model_trans  # rng, state, action -> state
        self.model_obs = model_obs  # rng, state -> obs
        self.model_rew = model_rew  # rng, state -> R
        self.model_done = model_done  # rng, state -> done

    def sample_params(self, rng):
        _rng_init, _rng_trans, _rng_obs, _rng_rew, _rng_done = jax.random.split(rng, 5)
        init = self.model_init.sample_params(_rng_init)
        trans = self.model_trans.sample_params(_rng_trans)
        obs = self.model_obs.sample_params(_rng_obs)
        rew = self.model_rew.sample_params(_rng_rew)
        done = self.model_done.sample_params(_rng_done)
        return dict(init=init, trans=trans, obs=obs, rew=rew, done=done)

    def reset_env(self, rng, params):
        """Performs resetting of environment."""
        _rng_init, _rng_obs = split(rng)
        state = self.model_init(_rng_init, params['init'])
        obs = self.model_obs(_rng_obs, state, params['obs'])
        return obs, state

    def step_env(self, rng, state, action, params):
        """Performs step transitions in the environment."""
        _rng_trans, _rng_obs, _rng_rew, _rng_done = jax.random.split(rng, 4)
        state = self.model_trans(_rng_trans, state, action, params['trans'])
        obs = self.model_obs(_rng_obs, state, params['obs'])
        rew = self.model_rew(_rng_rew, state, params['rew'])
        done = self.model_done(_rng_done, state, params['done'])
        info = {}
        return obs, state, rew, done, info

    def get_rew(self, rng, state, params):
        return self.model_rew(rng, state, params['rew'])

    def is_done(self, rng, state, params):
        return self.model_done(rng, state, params['done'])

    @property
    def name(self) -> str:
        return "SyntheticMDP"

    def action_space(self, params):
        # return self.model_trans.action_space(params['params_trans']) #TODO look into this...
        return self.model_trans.action_space(None)

    def observation_space(self, params):
        return self.model_obs.observation_space(params['obs'])

    def state_space(self, params):
        return self.model_init.state_space(params['init'])

    def get_obs(self, state, params):
        raise NotImplementedError

    @property
    def num_actions(self) -> int:
        raise NotImplementedError


class IdentityObs:
    def sample_params(self, rng):
        return None

    def __call__(self, rng, state, params):
        return state

    def observation_space(self, params):
        return None


class NeverDone:
    def sample_params(self, rng):
        return None

    def __call__(self, rng, state, params):
        return False


class Discrete(spaces.Discrete):
    def sample(self, rng):
        return jax.random.randint(rng, (), minval=0, maxval=self.n, dtype=int)


class Box(spaces.Box):
    def sample(self, rng):
        return jax.random.uniform(rng, self.shape, minval=self.low, maxval=self.high, dtype=None)
