#!/bin/bash
source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps/bin/activate
cd /data/vision/phillipi/akumar01/synthetic-mdps/src



CUDA_VISIBLE_DEVICES=0 python icl_bc.py --dataset_paths="../data/exp_icl//datasets/real/*/*/dataset.pkl"                                                                           --exclude_dataset_paths=None                                                                              --save_dir="../data/exp_icl//train_bc/all"                                                                    --n_iters=2000 --save_agent=True &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --dataset_paths="../data/exp_icl//datasets/real/*/*/dataset.pkl"                                                                           --exclude_dataset_paths="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --save_dir="../data/exp_icl//train_bc/all-name=CartPole-v1"                                                   --n_iters=2000 --save_agent=True &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --dataset_paths="../data/exp_icl//datasets/real/*/*/dataset.pkl"                                                                           --exclude_dataset_paths="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --save_dir="../data/exp_icl//train_bc/all-name=Acrobot-v1"                                                    --n_iters=2000 --save_agent=True &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --dataset_paths="../data/exp_icl//datasets/real/*/*/dataset.pkl"                                                                           --exclude_dataset_paths="../data/exp_icl//datasets//real/classic/name=MountainCar-v0//dataset.pkl"        --save_dir="../data/exp_icl//train_bc/all-name=MountainCar-v0"                                                --n_iters=2000 --save_agent=True &
wait
