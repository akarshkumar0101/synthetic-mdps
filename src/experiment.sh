#!/bin/bash
source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps/bin/activate
cd /data/vision/phillipi/akumar01/synthetic-mdps/src





CUDA_VISIBLE_DEVICES=0 python run.py --env_id="name=csmdp;d_state=2;d_obs=4;n_acts=2;delta=T;trans=linear;rew=linear;mrl=4x64" --agent_id="obs_embed=dense;name=linear_transformer;tl=256" --save_dir="../data/exp_01_17//pretrain/name=csmdp;d_state=2;d_obs=4;n_acts=2;delta=T;trans=linear;rew=linear;mrl=4x64" --save_agent_params=True --n_iters=10000 --n_envs=128 --n_envs_batch=32 &
CUDA_VISIBLE_DEVICES=1 python run.py --env_id="name=csmdp;d_state=2;d_obs=4;n_acts=2;delta=T;trans=linear;rew=goal;mrl=4x64"   --agent_id="obs_embed=dense;name=linear_transformer;tl=256" --save_dir="../data/exp_01_17//pretrain/name=csmdp;d_state=2;d_obs=4;n_acts=2;delta=T;trans=linear;rew=goal;mrl=4x64"   --save_agent_params=True --n_iters=10000 --n_envs=128 --n_envs_batch=32 &
CUDA_VISIBLE_DEVICES=2 python run.py --env_id="name=csmdp;d_state=2;d_obs=4;n_acts=2;delta=T;trans=mlp;rew=linear;mrl=4x64"    --agent_id="obs_embed=dense;name=linear_transformer;tl=256" --save_dir="../data/exp_01_17//pretrain/name=csmdp;d_state=2;d_obs=4;n_acts=2;delta=T;trans=mlp;rew=linear;mrl=4x64"    --save_agent_params=True --n_iters=10000 --n_envs=128 --n_envs_batch=32 &
CUDA_VISIBLE_DEVICES=3 python run.py --env_id="name=csmdp;d_state=2;d_obs=4;n_acts=2;delta=T;trans=mlp;rew=goal;mrl=4x64"      --agent_id="obs_embed=dense;name=linear_transformer;tl=256" --save_dir="../data/exp_01_17//pretrain/name=csmdp;d_state=2;d_obs=4;n_acts=2;delta=T;trans=mlp;rew=goal;mrl=4x64"      --save_agent_params=True --n_iters=10000 --n_envs=128 --n_envs_batch=32 &
CUDA_VISIBLE_DEVICES=4 python run.py --env_id="name=csmdp;d_state=4;d_obs=4;n_acts=2;delta=T;trans=linear;rew=linear;mrl=4x64" --agent_id="obs_embed=dense;name=linear_transformer;tl=256" --save_dir="../data/exp_01_17//pretrain/name=csmdp;d_state=4;d_obs=4;n_acts=2;delta=T;trans=linear;rew=linear;mrl=4x64" --save_agent_params=True --n_iters=10000 --n_envs=128 --n_envs_batch=32 &
CUDA_VISIBLE_DEVICES=5 python run.py --env_id="name=csmdp;d_state=4;d_obs=4;n_acts=2;delta=T;trans=linear;rew=goal;mrl=4x64"   --agent_id="obs_embed=dense;name=linear_transformer;tl=256" --save_dir="../data/exp_01_17//pretrain/name=csmdp;d_state=4;d_obs=4;n_acts=2;delta=T;trans=linear;rew=goal;mrl=4x64"   --save_agent_params=True --n_iters=10000 --n_envs=128 --n_envs_batch=32 &
wait

CUDA_VISIBLE_DEVICES=0 python run.py --n_seeds=4 --env_id="name=CartPole-v1;tl=256" --agent_id="obs_embed=dense;name=linear_transformer;tl=256" --save_dir="../data/exp_01_17//expert/name=CartPole-v1;tl=256" --save_agent_params=True --n_iters=1000 --n_envs=128 --n_envs_batch=32 &
wait

CUDA_VISIBLE_DEVICES=0 python run_bc.py --n_seeds=32 --env_id="name=CartPole-v1;tl=256" --agent_id="obs_embed=dense;name=linear_transformer;tl=256" --load_dir="../data/exp_01_17//pretrain/name=csmdp;d_state=2;d_obs=4;n_acts=2;delta=T;trans=linear;rew=linear;mrl=4x64" --load_dir_teacher="../data/exp_01_17//expert/name=CartPole-v1;tl=256" --save_dir="../data/exp_01_17//test/name=CartPole-v1;tl=256/name=csmdp;d_state=2;d_obs=4;n_acts=2;delta=T;trans=linear;rew=linear;mrl=4x64" --reset_layers="last" --ft_layers="last" --n_iters=300 --n_envs_batch=4 --lr=0.00025 &
CUDA_VISIBLE_DEVICES=1 python run_bc.py --n_seeds=32 --env_id="name=CartPole-v1;tl=256" --agent_id="obs_embed=dense;name=linear_transformer;tl=256" --load_dir="../data/exp_01_17//pretrain/name=csmdp;d_state=2;d_obs=4;n_acts=2;delta=T;trans=linear;rew=goal;mrl=4x64"   --load_dir_teacher="../data/exp_01_17//expert/name=CartPole-v1;tl=256" --save_dir="../data/exp_01_17//test/name=CartPole-v1;tl=256/name=csmdp;d_state=2;d_obs=4;n_acts=2;delta=T;trans=linear;rew=goal;mrl=4x64"   --reset_layers="last" --ft_layers="last" --n_iters=300 --n_envs_batch=4 --lr=0.00025 &
CUDA_VISIBLE_DEVICES=2 python run_bc.py --n_seeds=32 --env_id="name=CartPole-v1;tl=256" --agent_id="obs_embed=dense;name=linear_transformer;tl=256" --load_dir="../data/exp_01_17//pretrain/name=csmdp;d_state=2;d_obs=4;n_acts=2;delta=T;trans=mlp;rew=linear;mrl=4x64"    --load_dir_teacher="../data/exp_01_17//expert/name=CartPole-v1;tl=256" --save_dir="../data/exp_01_17//test/name=CartPole-v1;tl=256/name=csmdp;d_state=2;d_obs=4;n_acts=2;delta=T;trans=mlp;rew=linear;mrl=4x64"    --reset_layers="last" --ft_layers="last" --n_iters=300 --n_envs_batch=4 --lr=0.00025 &
CUDA_VISIBLE_DEVICES=3 python run_bc.py --n_seeds=32 --env_id="name=CartPole-v1;tl=256" --agent_id="obs_embed=dense;name=linear_transformer;tl=256" --load_dir="../data/exp_01_17//pretrain/name=csmdp;d_state=2;d_obs=4;n_acts=2;delta=T;trans=mlp;rew=goal;mrl=4x64"      --load_dir_teacher="../data/exp_01_17//expert/name=CartPole-v1;tl=256" --save_dir="../data/exp_01_17//test/name=CartPole-v1;tl=256/name=csmdp;d_state=2;d_obs=4;n_acts=2;delta=T;trans=mlp;rew=goal;mrl=4x64"      --reset_layers="last" --ft_layers="last" --n_iters=300 --n_envs_batch=4 --lr=0.00025 &
CUDA_VISIBLE_DEVICES=4 python run_bc.py --n_seeds=32 --env_id="name=CartPole-v1;tl=256" --agent_id="obs_embed=dense;name=linear_transformer;tl=256" --load_dir="../data/exp_01_17//pretrain/name=csmdp;d_state=4;d_obs=4;n_acts=2;delta=T;trans=linear;rew=linear;mrl=4x64" --load_dir_teacher="../data/exp_01_17//expert/name=CartPole-v1;tl=256" --save_dir="../data/exp_01_17//test/name=CartPole-v1;tl=256/name=csmdp;d_state=4;d_obs=4;n_acts=2;delta=T;trans=linear;rew=linear;mrl=4x64" --reset_layers="last" --ft_layers="last" --n_iters=300 --n_envs_batch=4 --lr=0.00025 &
CUDA_VISIBLE_DEVICES=5 python run_bc.py --n_seeds=32 --env_id="name=CartPole-v1;tl=256" --agent_id="obs_embed=dense;name=linear_transformer;tl=256" --load_dir="../data/exp_01_17//pretrain/name=csmdp;d_state=4;d_obs=4;n_acts=2;delta=T;trans=linear;rew=goal;mrl=4x64"   --load_dir_teacher="../data/exp_01_17//expert/name=CartPole-v1;tl=256" --save_dir="../data/exp_01_17//test/name=CartPole-v1;tl=256/name=csmdp;d_state=4;d_obs=4;n_acts=2;delta=T;trans=linear;rew=goal;mrl=4x64"   --reset_layers="last" --ft_layers="last" --n_iters=300 --n_envs_batch=4 --lr=0.00025 &
wait
CUDA_VISIBLE_DEVICES=0 python run_bc.py --n_seeds=32 --env_id="name=CartPole-v1;tl=256" --agent_id="obs_embed=dense;name=linear_transformer;tl=256" --load_dir=None                                                                                                         --load_dir_teacher="../data/exp_01_17//expert/name=CartPole-v1;tl=256" --save_dir="../data/exp_01_17//test/name=CartPole-v1;tl=256/random_agent"                                                                   --reset_layers="last" --ft_layers="last" --n_iters=300 --n_envs_batch=4 --lr=0.0     &
CUDA_VISIBLE_DEVICES=1 python run_bc.py --n_seeds=32 --env_id="name=CartPole-v1;tl=256" --agent_id="obs_embed=dense;name=linear_transformer;tl=256" --load_dir=None                                                                                                         --load_dir_teacher="../data/exp_01_17//expert/name=CartPole-v1;tl=256" --save_dir="../data/exp_01_17//test/name=CartPole-v1;tl=256/train_random"                                                                   --reset_layers="last" --ft_layers="last" --n_iters=300 --n_envs_batch=4 --lr=0.00025 &
wait
