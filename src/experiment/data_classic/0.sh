#!/bin/bash
source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps/bin/activate
cd /data/vision/phillipi/akumar01/synthetic-mdps/src



CUDA_VISIBLE_DEVICES=0 python icl_gen_ed.py --env_id="name=CartPole-v1;tl=500"         --agent_id="classic" --save_dir="/data/vision/phillipi/akumar01/synthetic-mdps-data/datasets//real/classic/name=CartPole-v1;tl=500//"         --best_of_n_experts=50 --n_iters_train=4000 --n_iters_eval=64 --n_envs=128 --lr=0.0003 &
CUDA_VISIBLE_DEVICES=1 python icl_gen_ed.py --env_id="name=Acrobot-v1;tl=500"          --agent_id="classic" --save_dir="/data/vision/phillipi/akumar01/synthetic-mdps-data/datasets//real/classic/name=Acrobot-v1;tl=500//"          --best_of_n_experts=50 --n_iters_train=4000 --n_iters_eval=64 --n_envs=128 --lr=0.0003 &
CUDA_VISIBLE_DEVICES=2 python icl_gen_ed.py --env_id="name=MountainCar-v0;tl=500"      --agent_id="classic" --save_dir="/data/vision/phillipi/akumar01/synthetic-mdps-data/datasets//real/classic/name=MountainCar-v0;tl=500//"      --best_of_n_experts=50 --n_iters_train=4000 --n_iters_eval=64 --n_envs=128 --lr=0.0003 &
CUDA_VISIBLE_DEVICES=3 python icl_gen_ed.py --env_id="name=DiscretePendulum-v1;tl=500" --agent_id="classic" --save_dir="/data/vision/phillipi/akumar01/synthetic-mdps-data/datasets//real/classic/name=DiscretePendulum-v1;tl=500//" --best_of_n_experts=50 --n_iters_train=4000 --n_iters_eval=64 --n_envs=128 --lr=0.0003 &
wait
