#!/bin/bash
source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps/bin/activate
cd /data/vision/phillipi/akumar01/synthetic-mdps/src



CUDA_VISIBLE_DEVICES=0 python icl_gen.py --env_id="name=CartPole-v1"         --agent_id="classic" --save_dir="../data/exp_icl//datasets//real/classic/name=CartPole-v1//"         --best_of_n_experts=20 --n_iters_train=4000 --n_iters_eval=1024 --lr=0.0003 &
CUDA_VISIBLE_DEVICES=1 python icl_gen.py --env_id="name=Acrobot-v1"          --agent_id="classic" --save_dir="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//"          --best_of_n_experts=20 --n_iters_train=4000 --n_iters_eval=1024 --lr=0.0003 &
CUDA_VISIBLE_DEVICES=2 python icl_gen.py --env_id="name=MountainCar-v0"      --agent_id="classic" --save_dir="../data/exp_icl//datasets//real/classic/name=MountainCar-v0//"      --best_of_n_experts=20 --n_iters_train=4000 --n_iters_eval=1024 --lr=0.0003 &
CUDA_VISIBLE_DEVICES=3 python icl_gen.py --env_id="name=DiscretePendulum-v1" --agent_id="classic" --save_dir="../data/exp_icl//datasets//real/classic/name=DiscretePendulum-v1//" --best_of_n_experts=20 --n_iters_train=4000 --n_iters_eval=1024 --lr=0.0003 &
wait