#!/bin/bash
source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps/bin/activate
cd /data/vision/phillipi/akumar01/synthetic-mdps/src





CUDA_VISIBLE_DEVICES=0 python icl_gen.py --env_id="name=csmdp;d_state=2;d_obs=4;n_acts=4;delta=T;trans=linear;rew=goal;tl=64"   --save_dir="../data/exp_iclbc//datasets/name=csmdp;d_state=2;d_obs=4;n_acts=4;delta=T;trans=linear;rew=goal;tl=64/"   --n_seeds_seq=16 --n_seeds_par=16 --n_iters_train=100 --lr=0.0003 &
CUDA_VISIBLE_DEVICES=1 python icl_gen.py --env_id="name=csmdp;d_state=2;d_obs=4;n_acts=4;delta=T;trans=linear;rew=linear;tl=64" --save_dir="../data/exp_iclbc//datasets/name=csmdp;d_state=2;d_obs=4;n_acts=4;delta=T;trans=linear;rew=linear;tl=64/" --n_seeds_seq=16 --n_seeds_par=16 --n_iters_train=100 --lr=0.0003 &
wait

CUDA_VISIBLE_DEVICES=0 python icl_gen.py --env_id="name=CartPole-v1"         --agent_id="classic" --save_dir="../data/exp_iclbc//datasets/name=CartPole-v1/"         --best_of_n_experts=10 --n_iters_train=2000 --n_iters_eval=300 --lr=0.0003 &
CUDA_VISIBLE_DEVICES=1 python icl_gen.py --env_id="name=Acrobot-v1"          --agent_id="classic" --save_dir="../data/exp_iclbc//datasets/name=Acrobot-v1/"          --best_of_n_experts=10 --n_iters_train=2000 --n_iters_eval=300 --lr=0.0003 &
CUDA_VISIBLE_DEVICES=2 python icl_gen.py --env_id="name=MountainCar-v0"      --agent_id="classic" --save_dir="../data/exp_iclbc//datasets/name=MountainCar-v0/"      --best_of_n_experts=10 --n_iters_train=2000 --n_iters_eval=300 --lr=0.0003 &
CUDA_VISIBLE_DEVICES=3 python icl_gen.py --env_id="name=DiscretePendulum-v1" --agent_id="classic" --save_dir="../data/exp_iclbc//datasets/name=DiscretePendulum-v1/" --best_of_n_experts=10 --n_iters_train=2000 --n_iters_eval=300 --lr=0.0003 &
wait

CUDA_VISIBLE_DEVICES=0 python icl_gen.py --env_id="name=Asterix-MinAtar"       --agent_id="minatar" --save_dir="../data/exp_iclbc//datasets/name=Asterix-MinAtar/"       --best_of_n_experts=10 --n_iters_train=2000 --n_iters_eval=20 --n_envs=64 --n_updates=32 --n_envs_batch=8 --lr=0.001 --gamma=0.999 &
CUDA_VISIBLE_DEVICES=1 python icl_gen.py --env_id="name=Breakout-MinAtar"      --agent_id="minatar" --save_dir="../data/exp_iclbc//datasets/name=Breakout-MinAtar/"      --best_of_n_experts=10 --n_iters_train=2000 --n_iters_eval=20 --n_envs=64 --n_updates=32 --n_envs_batch=8 --lr=0.001 --gamma=0.999 &
CUDA_VISIBLE_DEVICES=2 python icl_gen.py --env_id="name=Freeway-MinAtar"       --agent_id="minatar" --save_dir="../data/exp_iclbc//datasets/name=Freeway-MinAtar/"       --best_of_n_experts=10 --n_iters_train=2000 --n_iters_eval=20 --n_envs=64 --n_updates=32 --n_envs_batch=8 --lr=0.001 --gamma=0.999 &
CUDA_VISIBLE_DEVICES=3 python icl_gen.py --env_id="name=SpaceInvaders-MinAtar" --agent_id="minatar" --save_dir="../data/exp_iclbc//datasets/name=SpaceInvaders-MinAtar/" --best_of_n_experts=10 --n_iters_train=2000 --n_iters_eval=20 --n_envs=64 --n_updates=32 --n_envs_batch=8 --lr=0.001 --gamma=0.999 &
wait
