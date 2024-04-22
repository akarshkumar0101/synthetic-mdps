from functools import partial

import distrax
import jax
import jax.numpy as jnp
import optax
from einops import rearrange
from jax.random import split

from .ppo_dr import PPO


def kl_div(logits, logits_target, axis=-1):
    log_p, log_q = jax.nn.log_softmax(logits_target), jax.nn.log_softmax(logits)
    return (jnp.exp(log_p) * (log_p - log_q)).sum(axis=axis)


class BC:
    def __init__(self, agent_student, agent_teacher, agent_params_teacher, env, sample_env_params, tx=None, *,
                 n_envs=4, n_steps=128, n_updates=16, n_envs_batch=1, lr=2.5e-4, clip_grad_norm=0.5,
                 ent_coef=0.0):
        self.agent_student, self.agent_teacher = agent_student, agent_teacher
        self.agent_params_teacher = agent_params_teacher
        self.env = env
        self.sample_env_params = sample_env_params
        self.config = dict(n_envs=n_envs, n_steps=n_steps, n_updates=n_updates, n_envs_batch=n_envs_batch,
                           lr=lr, clip_grad_norm=clip_grad_norm, ent_coef=ent_coef)
        self.n_envs, self.n_steps = n_envs, n_steps
        self.n_updates, self.n_envs_batch = n_updates, n_envs_batch
        self.lr, self.clip_grad_norm = lr, clip_grad_norm
        self.ent_coef = ent_coef

        if tx is None:
            tx = optax.chain(optax.clip_by_global_norm(self.clip_grad_norm), optax.adam(self.lr, eps=1e-5))
        self.tx = tx

        self.ppo_student = PPO(agent_student, env, sample_env_params, tx=tx, n_envs=n_envs, n_steps=n_steps,
                               n_updates=0)
        self.ppo_teacher = PPO(agent_teacher, env, sample_env_params, tx=None, n_envs=n_envs, n_steps=n_steps,
                               n_updates=0)

    def loss_fn(self, agent_params, agent_state_student, batch_teacher):
        forward_parallel = partial(self.agent_student.apply, method=self.agent_student.forward_parallel)
        obs = batch_teacher['obs']
        _, (logits, val) = jax.vmap(forward_parallel, in_axes=(None, 0, 0))(agent_params, agent_state_student, obs)
        pi = distrax.Categorical(logits=logits)
        loss_actor = optax.softmax_cross_entropy_with_integer_labels(logits, batch_teacher['act']).mean()
        entropy = pi.entropy().mean()
        loss = 1.0 * loss_actor - self.ent_coef * entropy
        return loss, (loss_actor, entropy)

    def update_batch(self, carry, _):
        rng, train_state, agent_state_student, buffer_teacher = carry

        rng, _rng = split(rng)
        idx_env = jax.random.permutation(_rng, self.n_envs)[:self.n_envs_batch]
        batch = jax.tree_util.tree_map(lambda x: x[:, idx_env], buffer_teacher)
        batch = jax.tree_util.tree_map(lambda x: rearrange(x, 'T B ... -> B T ...'), batch)
        agent_state_student_batch = jax.tree_map(lambda x: x[idx_env], agent_state_student)

        grad_fn = jax.value_and_grad(self.loss_fn, has_aux=True)
        ((loss, (loss_actor, entropy)), grads) = grad_fn(train_state.params, agent_state_student_batch, batch)
        train_state = train_state.apply_gradients(grads=grads)

        carry = rng, train_state, agent_state_student, buffer_teacher
        return carry, loss

    def bc_step(self, carry, _):
        agent_state_student, carry_student, carry_teacher = carry

        carry_student, buffer_student = self.ppo_student.eval_step(carry_student, None)
        carry_teacher, buffer_teacher = self.ppo_teacher.eval_step(carry_teacher, None)

        rng, train_state, env_params, agent_state, obs, env_state = carry_student

        carry = rng, train_state, agent_state_student, buffer_teacher
        carry, losses = jax.lax.scan(self.update_batch, carry, None, self.n_updates)
        rng, train_state, agent_state_student, buffer_teacher = carry

        carry_student = rng, train_state, env_params, agent_state, obs, env_state

        # update the agent_state_student
        forward_parallel = partial(self.agent_student.apply, method=self.agent_student.forward_parallel)
        obs = jax.tree_util.tree_map(lambda x: rearrange(x, 'T B ... -> B T ...'), buffer_teacher['obs'])
        agent_state_student, _ = jax.vmap(forward_parallel, in_axes=(None, 0, 0))(train_state.params,
                                                                                  agent_state_student, obs)

        carry = agent_state_student, carry_student, carry_teacher
        return carry, (buffer_student, buffer_teacher, losses)

    def init_agent_env(self, rng):
        carry_student = self.ppo_student.init_agent_env(rng)
        carry_teacher = self.ppo_teacher.init_agent_env(rng)  # okay to use same rng

        rng, train_state, env_params, agent_state, obs, env_state = carry_teacher
        train_state = train_state.replace(params=self.agent_params_teacher)
        carry_teacher = rng, train_state, env_params, agent_state, obs, env_state

        _, _, _, agent_state_student, _, _ = carry_student
        return agent_state_student, carry_student, carry_teacher
