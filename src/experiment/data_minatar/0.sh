#!/bin/bash
source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps/bin/activate
cd /data/vision/phillipi/akumar01/synthetic-mdps/src



CUDA_VISIBLE_DEVICES=0 python icl_gen_ed.py --env_id="name=Asterix-MinAtar"       --agent_id="minatar" --save_dir="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//"       --best_of_n_experts=10 --n_iters_train=2000 --n_iters_eval=128 --n_envs=64 --n_updates=32 --n_envs_batch=8 --lr=0.001 --gamma=0.999 &
CUDA_VISIBLE_DEVICES=1 python icl_gen_ed.py --env_id="name=Breakout-MinAtar"      --agent_id="minatar" --save_dir="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//"      --best_of_n_experts=10 --n_iters_train=2000 --n_iters_eval=128 --n_envs=64 --n_updates=32 --n_envs_batch=8 --lr=0.001 --gamma=0.999 &
CUDA_VISIBLE_DEVICES=2 python icl_gen_ed.py --env_id="name=Freeway-MinAtar"       --agent_id="minatar" --save_dir="../data/exp_icl//datasets//real/minatar/name=Freeway-MinAtar//"       --best_of_n_experts=10 --n_iters_train=2000 --n_iters_eval=128 --n_envs=64 --n_updates=32 --n_envs_batch=8 --lr=0.001 --gamma=0.999 &
CUDA_VISIBLE_DEVICES=3 python icl_gen_ed.py --env_id="name=SpaceInvaders-MinAtar" --agent_id="minatar" --save_dir="../data/exp_icl//datasets//real/minatar/name=SpaceInvaders-MinAtar//" --best_of_n_experts=10 --n_iters_train=2000 --n_iters_eval=128 --n_envs=64 --n_updates=32 --n_envs_batch=8 --lr=0.001 --gamma=0.999 &
wait
