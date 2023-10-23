import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
import distrax
import gymnax
from wrappers import LogWrapper, FlattenObservationWrapper
from einops import rearrange
from functools import partial

from agents.basic import BasicAgent


def collect(rng, env, env_params, env_state, forward_recurrent, agent_params, agent_state, oar, n_envs, n_steps):
    def _env_step(runner_state, _):
        rng, env_state, agent_state, (obs, act_p, rew_p) = runner_state

        # SELECT ACTION
        agent_state_n, (logits, value) = forward_recurrent(agent_params, agent_state, (obs, act_p, rew_p))
        rng, _rng = jax.random.split(rng)
        pi = distrax.Categorical(logits=logits)
        act = pi.sample(seed=_rng)
        log_prob = pi.log_prob(act)

        # STEP ENV
        rng, _rng = jax.random.split(rng)
        _rng = jax.random.split(_rng, n_envs)
        obs_n, env_state_n, rew, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(_rng, env_state, act,
                                                                                          env_params)
        runner_state = rng, env_state_n, agent_state_n, (obs_n, act, rew)
        transition = dict(obs=obs, act=act, rew=rew, done=done, info=info,
                          act_p=act_p, rew_p=rew_p,
                          logits=logits, log_prob=log_prob, val=value)
        return runner_state, transition

    runner_state = rng, env_state, agent_state, oar
    runner_state, buffer = jax.lax.scan(_env_step, runner_state, None, n_steps)
    return buffer, env_state, agent_state, oar


def calc_gae(buffer, last_val=None, gamma=0.99, gae_lambda=0.95):
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        done, value, reward = transition["done"], transition["val"], transition["rew"]
        delta = reward + gamma * next_value * (1 - done) - value
        gae = (delta + gamma * gae_lambda * (1 - done) * gae)
        return (gae, value), gae

    if last_val is None:
        last_val = buffer['val'][-1, :]
    _, adv = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val), buffer, reverse=True, unroll=16)
    return adv, adv + buffer['val']


def make_train(config):
    config["NUM_UPDATES"] = (config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
    config["MINIBATCH_SIZE"] = (config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"])
    env, env_params = gymnax.make(config["ENV_NAME"])
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = (1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"])
        return config["LR"] * frac

    def train(rng):
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        _rng = jax.random.split(_rng, config["NUM_ENVS"])
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(_rng, env_params)
        act_p, rew_p = jnp.zeros(config["NUM_ENVS"], dtype=jnp.int32), jnp.zeros(config["NUM_ENVS"])
        oar = obs, act_p, rew_p

        # INIT NETWORK
        network = BasicAgent(env.action_space(env_params).n, activation=config["ACTIVATION"])
        forward_recurrent = jax.vmap(partial(network.apply, method=network.forward_recurrent),
                                     in_axes=(None, 0, (0, 0, 0)))
        forward_parallel = jax.vmap(partial(network.apply, method=network.forward_parallel),
                                    in_axes=(None, 0, 0, 0))

        rng, _rng = jax.random.split(rng)
        _rng = jax.random.split(_rng, config["NUM_ENVS"])
        agent_init_state = jax.vmap(network.get_init_state)(_rng)

        rng, _rng = jax.random.split(rng)
        network_params = network.init(_rng, agent_init_state[0], (obs[0], act_p[0], rew_p[0]),
                                      method=network.forward_recurrent)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx, )

        # TRAIN LOOP
        def _update_step(runner_state, _):
            train_state, env_state, agent_state, oar, rng = runner_state

            # COLLECT TRAJECTORIES
            rng, _rng = jax.random.split(rng)
            traj_batch, env_state, agent_state, oar = collect(_rng, env, env_params, env_state,
                                        forward_recurrent, train_state.params, agent_state,
                                        oar, n_envs=config["NUM_ENVS"], n_steps=config["NUM_STEPS"])

            # CALCULATE ADVANTAGE
            traj_batch['adv'], traj_batch['ret'] = calc_gae(traj_batch, last_val=None,
                                                            gamma=config["GAMMA"], gae_lambda=config["GAE_LAMBDA"])

            # runner_state = train_state, env_state, agent_state, oar, rng

            # UPDATE NETWORK
            def _update_batch(update_state, _):
                train_state, traj_batch, rng = update_state
                batch = traj_batch

                rng, _rng = jax.random.split(rng)
                n_envs_batch = config["NUM_ENVS"] // config["NUM_MINIBATCHES"]
                idx_env = jax.random.permutation(_rng, config["NUM_ENVS"])[:n_envs_batch]
                batch = jax.tree_util.tree_map(lambda x: x[:, idx_env], batch)
                # batch = jax.tree_util.tree_map( lambda x: rearrange(x, 't n ... -> (t n) ...'), batch )
                batch = jax.tree_util.tree_map(lambda x: rearrange(x, 't n ... -> n t ...'), batch)

                def _loss_fn(params, batch):
                    # RERUN NETWORK
                    logits, value = forward_parallel(params, batch['obs'], batch['act_p'], batch['rew_p'])
                    pi = distrax.Categorical(logits=logits)
                    log_prob = pi.log_prob(batch['act'])

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = batch['val'] + (value - batch['val']).clip(-config["CLIP_EPS"],
                                                                                    config["CLIP_EPS"])
                    value_losses = jnp.square(value - batch['ret'])
                    value_losses_clipped = jnp.square(value_pred_clipped - batch['ret'])
                    value_loss = (0.5 * jnp.maximum(value_losses, value_losses_clipped).mean())

                    # CALCULATE ACTOR LOSS
                    ratio = jnp.exp(log_prob - batch['log_prob'])
                    gae = batch['adv']
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"], ) * gae)
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()
                    entropy = pi.entropy().mean()

                    total_loss = (loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy)
                    return total_loss, (value_loss, loss_actor, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(train_state.params, batch)
                train_state = train_state.apply_gradients(grads=grads)

                update_state = train_state, traj_batch, rng
                return update_state, total_loss

            update_state = (train_state, traj_batch, rng)
            update_state, loss_info = jax.lax.scan(_update_batch, update_state, None,
                                                   config["UPDATE_EPOCHS"] * config["NUM_MINIBATCHES"])
            train_state = update_state[0]
            metric = traj_batch['info']
            rng = update_state[-1]
            if config.get("DEBUG"):
                def callback(info):
                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    timesteps = info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    for t in range(len(timesteps)):
                        print(f"global step={timesteps[t]}, episodic return={return_values[t]}")

                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, agent_state, oar, rng)
            return runner_state, metric

        agent_state = jax.tree_map(lambda x: x, agent_init_state)  # copy
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, agent_state, oar, _rng)
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metric}

    return train


def main():
    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 4,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 2e5,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ENV_NAME": "CartPole-v1",
        "ANNEAL_LR": False,
        "DEBUG": False,
    }
    rng = jax.random.PRNGKey(30)
    train_fn = jax.jit(jax.vmap(make_train(config)))
    rng, *_rng = jax.random.split(rng, 1 + 4)
    out = train_fn(jnp.stack(_rng))
    metrics = out["metrics"]
    print(jax.tree_map(lambda x: x.shape, metrics))
    rets = metrics["returned_episode_returns"]  # n_seed, n_iters, n_steps, n_envs
    import matplotlib.pyplot as plt

    n_iters = rets.shape[1]
    steps = jnp.arange(n_iters) * config["NUM_STEPS"] * config["NUM_ENVS"]
    plt.plot(steps, jnp.mean(rets, axis=(0, 2, 3)), label='mean')
    plt.plot(steps, jnp.median(rets, axis=(0, 2, 3)), label='median')
    plt.plot(steps, jnp.mean(rets, axis=(2, 3)).T, c='gray', alpha=0.1)
    plt.legend()
    plt.ylabel('Return')
    plt.xlabel('Env Steps')
    plt.show()

if __name__ == '__main__':
    main()
