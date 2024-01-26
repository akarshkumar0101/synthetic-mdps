import argparse
import pickle
from functools import partial

import gymnax
import jax
import matplotlib.pyplot as plt
from einops import rearrange
from jax.random import split
from tqdm.auto import tqdm

import util
from agents.basic import BasicAgentSeparate, BigBasicAgentSeparate
from agents.util import DenseObsEmbed
from algos.ppo_dr import PPO
from mdps import csmdp
from mdps import smdp
from mdps.wrappers import LogWrapper
from mdps.wrappers_mine import TimeLimit, ObsNormRand, FlattenObservationWrapper

# parser = argparse.ArgumentParser()
# parser.add_argument("--env_id", type=str, default="CartPole-v1")
#
# parser.add_argument("--load_dir", type=str, default=None)
# parser.add_argument("--save_dir", type=str, default=None)
#
# parser.add_argument("--n_tasks", type=int, default=int(1e9))
# parser.add_argument("--n_iters", type=int, default=10000)
# parser.add_argument("--curriculum", type=str, default="none")
#
# parser.add_argument("--bs", type=int, default=256)
# parser.add_argument("--lr", type=float, default=2.5e-4)
# parser.add_argument("--clip_grad_norm", type=float, default=1.)
#
#
# # class MountainCarDenseRew(MyGymnaxWrapper):
# #
# #     def reset_env(self, key, params):
# #         obs, state = self._env.reset_env(key, params)
# #         return obs, state
# #
# #     def step_env(self, key, state, action, params):
# #         obs, state, rew, done, info = self._env.step_env(key, state, action, params)
# #         # pos_range = jnp.array([-1.2, 0.6])
# #         # vel_range = jnp.array([-0.07, 0.07])
# #         r = jnp.array([0.6 - -1.2, 0.07 - -0.07])
# #         mid = jnp.array([-jnp.pi / 6, 0.])
# #         a = jnp.array([state.position, state.velocity])
# #         a = ((a - mid) / r)
# #         a = jnp.linalg.norm(a)
# #         rew = a
# #         return obs, state, rew, done, info
#
# def generate_real_env_dataset(env_id):
#     rng = jax.random.PRNGKey(0)
#     env, env_params = gymnax.make(env_id)
#     env.sample_params = lambda rng: env_params
#     env = LogWrapper(env)
#     n_acts = env.action_space(env_params).n
#
#     ObsEmbed = partial(DenseObsEmbed, d_embd=128)
#     agent = BasicAgentSeparate(ObsEmbed, n_acts)
#
#     ppo = PPO(agent, env, sample_env_params=env.sample_params,
#               n_envs=4, n_steps=128, n_updates=16, n_envs_batch=1, lr=2.5e-4, clip_grad_norm=0.5,
#               clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, gamma=0.99, gae_lambda=0.95)
#     init_agent_env = jax.jit(jax.vmap(ppo.init_agent_env))
#     ppo_step = jax.jit(jax.vmap(ppo.ppo_step, in_axes=(0, None)))
#     eval_step = jax.jit(jax.vmap(ppo.eval_step, in_axes=(0, None)))
#
#     rng, _rng = split(rng)
#     carry = init_agent_env(split(_rng, 1))
#
#     rets = []
#     for i_iter in tqdm(range(100)):
#         carry, buffer = eval_step(carry, None)
#         # rets.append(buffer['info']['returned_episode_returns'].mean(axis=(-1, -2)))
#
#     for i_iter in tqdm(range(900)):
#         carry, buffer = ppo_step(carry, None)
#         rets.append(buffer['info']['returned_episode_returns'].mean(axis=(-1, -2)))
#
#     eval_stuff = []
#     for i_iter in tqdm(range(256 * 256)):
#         carry, buffer = eval_step(carry, None)
#         # rets.append(buffer['info']['returned_episode_returns'].mean(axis=(-1, -2)))
#
#         ks = ['obs', 'logits', 'act', 'rew', 'done']
#         eval_stuff.append({k: buffer[k] for k in ks})
#     rets = rearrange(jnp.stack(rets), 'N S -> S N')
#     # print(jax.tree_map(lambda x: x.shape, buffer))
#
#     eval_stuff = tree_stack(eval_stuff)
#     print(jax.tree_map(lambda x: x.shape, eval_stuff))
#     eval_stuff = jax.tree_map(lambda x: rearrange(x, 'N 1 T E ... -> (N E) T ...'), eval_stuff)  # only 0th seed
#     print(jax.tree_map(lambda x: x.shape, eval_stuff))
#
#     with open(f'../data/temp/expert_data_{env_id}.pkl', 'wb') as f:
#         pickle.dump(eval_stuff, f)
#
#     plt.plot(rets.T, c=[0.1, 0.1, 0.1, 0.1])
#     plt.plot(rets.mean(axis=0), label='mean')
#     plt.legend()
#     plt.ylabel('Return')
#     plt.xlabel('Training Iteration')
#
#     plt.show()
#
#
# def generate_syn_env_dataset():
#     env_id = "name=csmdp;d_state=2;d_obs=4;n_acts=4;delta=T;trans=linear;rew=goal;tl=64"
#
#     env = run.create_env(env_id)
#     ObsEmbed = partial(DenseObsEmbed, d_embd=32)
#     agent = BasicAgentSeparate(ObsEmbed, 4)
#
#     def get_dataset(rng):
#         rng, _rng = split(rng)
#         env_params = env.sample_params(_rng)
#
#         ppo = PPO(agent, env, sample_env_params=lambda rng: env_params,
#                   n_envs=4, n_steps=128, n_updates=16, n_envs_batch=1, lr=2.5e-4, clip_grad_norm=0.5,
#                   clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, gamma=0.99, gae_lambda=0.95)
#
#         rng, _rng = split(rng)
#         carry = ppo.init_agent_env(_rng)
#
#         carry, buffer = jax.lax.scan(ppo.ppo_step, carry, xs=None, length=100)
#         rets = buffer['info']['returned_episode_returns']
#
#         carry, buffer = jax.lax.scan(ppo.eval_step, carry, xs=None, length=1)
#         buffer = jax.tree_map(lambda x: rearrange(x, 'N T E ... -> (N E) T ...'), buffer)
#         dataset = {k: buffer[k] for k in ['obs', 'logits', 'act']}
#         return rets, dataset
#
#     rng = jax.random.PRNGKey(0)
#     rets, dataset = [], []
#     for _ in tqdm(range(6)):
#         rng, _rng = split(rng)
#         retsi, dataseti = jax.jit(jax.vmap(get_dataset))(split(rng, 64))
#         rets.append(retsi)
#         dataset.append(dataseti)
#     rets = util.tree_stack(rets)
#     dataset = util.tree_stack(dataset)
#     dataset = jax.tree_map(lambda x: rearrange(x, 'A B C T ... -> (A B C) T ...'), dataset)
#
#     print(rets.shape)
#     plt.plot(rets.mean(axis=(0, 1, -1, -2)))
#     plt.ylabel('Return')
#     plt.xlabel('Training Iteration')
#     plt.show()
#
#     print(jax.tree_map(lambda x: x.shape, dataset))
#     with open(f'../data/temp/expert_data_{"synthetic"}.pkl', 'wb') as f:
#         pickle.dump(dataset, f)


parser = argparse.ArgumentParser()
parser.add_argument("--env_id", type=str, default="CartPole-v1")
parser.add_argument("--agent_id", type=str, default="small")

parser.add_argument("--save_dir", type=str, default=None)

parser.add_argument("--best_of_n_experts", type=int, default=1)
parser.add_argument("--n_seeds_seq", type=int, default=1)  # sequential seeds
parser.add_argument("--n_seeds_par", type=int, default=1)  # parallel seeds

parser.add_argument("--n_iters_train", type=int, default=20)
parser.add_argument("--n_iters_eval", type=int, default=10)  # sequential envs
# ppo args
parser.add_argument("--n_envs", type=int, default=4)  # parallel envs
parser.add_argument("--n_steps", type=int, default=128)
parser.add_argument("--n_updates", type=int, default=16)
parser.add_argument("--n_envs_batch", type=int, default=1)
parser.add_argument("--lr", type=float, default=2.5e-4)
parser.add_argument("--clip_grad_norm", type=float, default=0.5)
parser.add_argument("--clip_eps", type=float, default=0.2)
parser.add_argument("--vf_coef", type=float, default=0.5)
parser.add_argument("--ent_coef", type=float, default=0.01)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae_lambda", type=float, default=0.95)


def create_env(env_id):
    env_cfg = dict([sub.split('=') for sub in env_id.split(';')])

    if env_cfg['name'] in gymnax.registered_envs:
        env, env_params = gymnax.make(env_cfg['name'])
        env.sample_params = lambda rng: env_params
    elif env_cfg['name'] == 'csmdp':
        d_state, d_obs, n_acts = int(env_cfg['d_state']), int(env_cfg['d_obs']), int(env_cfg['n_acts'])
        delta = env_cfg['delta'] == 'T'
        model_init = csmdp.Init(d_state=d_state)
        if env_cfg['trans'] == 'linear':
            model_trans = csmdp.LinearTransition(d_state=d_state, n_acts=n_acts, delta=delta)
        elif env_cfg['trans'] == 'mlp':
            model_trans = csmdp.MLPTransition(d_state=d_state, n_acts=n_acts, delta=delta)
        else:
            raise NotImplementedError
        model_obs = csmdp.LinearObservation(d_state=d_state, d_obs=d_obs)
        if env_cfg['rew'] == 'linear':
            model_rew = csmdp.LinearReward(d_state=d_state)
        elif env_cfg['rew'] == 'goal':
            model_rew = csmdp.GoalReward(d_state=d_state)
        else:
            raise NotImplementedError
        model_done = smdp.NeverDone()
        env = smdp.SyntheticMDP(model_init, model_trans, model_obs, model_rew, model_done)
    else:
        raise NotImplementedError
    env = FlattenObservationWrapper(env)
    # env = ObsNormRand(env)
    if 'tl' in env_cfg:
        env = TimeLimit(env, n_steps_max=int(env_cfg['tl']))
    env = LogWrapper(env)
    return env


def main(args):
    env = create_env(args.env_id)
    n_acts = env.action_space(None).n

    # ObsEmbed = partial(DenseObsEmbed, d_embd=(32 if args.agent_id == 'small' else 128))
    # agent = BasicAgentSeparate(ObsEmbed, n_acts)
    agent = BigBasicAgentSeparate(n_acts)

    def get_dataset_for_env_params(rng):
        rng, _rng = split(rng)
        env_params = env.sample_params(_rng)

        ppo = PPO(agent, env, sample_env_params=lambda rng: env_params,
                  n_envs=args.n_envs, n_steps=args.n_steps, n_updates=args.n_updates, n_envs_batch=args.n_envs_batch,
                  lr=args.lr, clip_grad_norm=args.clip_grad_norm, clip_eps=args.clip_eps,
                  vf_coef=args.vf_coef, ent_coef=args.ent_coef, gamma=args.gamma, gae_lambda=args.gae_lambda)

        init_agent_env_vmap = jax.vmap(ppo.init_agent_env)
        ppo_step_vmap = jax.vmap(ppo.ppo_step, in_axes=(0, None))

        rng, _rng = split(rng)
        carry = init_agent_env_vmap(split(_rng, args.best_of_n_experts))
        carry, buffer = jax.lax.scan(ppo_step_vmap, carry, xs=None, length=args.n_iters_train)
        rets_train = buffer['info']['returned_episode_returns']  # N S T E
        a = rets_train
        rets_train = rets_train.mean(axis=(-1, -2))
        idx_best_expert = rets_train[-1].argmax()
        rets_train = rets_train[:, idx_best_expert]
        # print(rets.shape, idx_best_expert)
        carry = jax.tree_map(lambda x: x[idx_best_expert], carry)

        carry, buffer = jax.lax.scan(ppo.eval_step, carry, xs=None, length=args.n_iters_eval)
        rets_eval = buffer['info']['returned_episode_returns'].mean()
        dataset = {k: buffer[k] for k in ['obs', 'logits', 'act']}
        # print(jax.tree_map(lambda x: x.shape, dataset))
        dataset = jax.tree_map(lambda x: rearrange(x, 'N T E ... -> (N E) T ...'), dataset)
        # print(jax.tree_map(lambda x: x.shape, dataset))
        # print(rets_eval.shape)
        return dataset, rets_train, rets_eval, None

    rng = jax.random.PRNGKey(0)
    data = []
    for _ in tqdm(range(args.n_seeds_seq)):
        rng, _rng = split(rng)
        dataset, rets_train, rets_eval, a = jax.jit(jax.vmap(get_dataset_for_env_params))(split(rng, args.n_seeds_par))
        data.append((dataset, rets_train, rets_eval))
    data = util.tree_stack(data)
    data = jax.tree_map(lambda x: rearrange(x, "S1 S2 ... -> (S1 S2) ..."), data)
    dataset, rets_train, rets_eval = data  # (S E T ...), (S N), (S )
    print(jax.tree_map(lambda x: x.shape, (dataset, rets_train, rets_eval)))

    plt.plot(rets_train.mean(axis=0), label='mean')
    n_seeds = args.n_seeds_seq * args.n_seeds_par
    plt.title(f'Training Curve for \n {env_id} \n({n_seeds} seeds, best of {args.best_of_n_experts} experts)')

    plt.ylabel('Return')
    plt.xlabel('Training Iteration')
    # plt.title(f'Final eval mean: {rets_eval.mean():6.3f}')
    plt.text(0.5, -0.2, f'Final eval mean: {rets_eval.mean():6.3f} +- {rets_eval.std(): 6.3f}', fontsize=30, color='r',
             transform=plt.gca().transAxes, horizontalalignment='center', verticalalignment='center')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # a = a[0] # (n_iters_train, n_experts, T, E)
    # a = rearrange(a, 'N S T E -> S E (N T)')
    # a = a[2]
    # plt.imshow(a)
    # plt.show()

    if args.save_dir is not None:
        with open(f'../data/temp/expert_data_{"synthetic"}.pkl', 'wb') as f:
            pickle.dump(dataset, f)


if __name__ == '__main__':
    # env_id = "name=CartPole-v1"
    # env_id = "name=Acrobot-v1"
    # env_id = "name=MountainCar-v0"
    # env_id = "name=Asterix-MinAtar"
    # env_id = "name=Breakout-MinAtar"
    # env_id = "name=Freeway-MinAtar"
    env_id = "name=SpaceInvaders-MinAtar"
    main(parser.parse_args(f"--env_id={env_id} --n_envs=64 --n_envs_batch=8 --n_updates=32 --gamma=.999 --n_iters_train=1000 --n_iters_eval=1 --lr=1e-3 --best_of_n_experts=5".split()))
