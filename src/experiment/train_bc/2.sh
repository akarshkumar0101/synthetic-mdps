#!/bin/bash
source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps/bin/activate
cd /data/vision/phillipi/akumar01/synthetic-mdps/src



CUDA_VISIBLE_DEVICES=0 python icl_bc.py --dataset_paths="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"                                                --exclude_dataset_paths=None                                                                              --save_dir="../data/exp_icl//train_bc/name=Asterix-MinAtar"                                                   --n_iters=2000 --save_agent=True &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --dataset_paths="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"                                               --exclude_dataset_paths=None                                                                              --save_dir="../data/exp_icl//train_bc/name=Breakout-MinAtar"                                                  --n_iters=2000 --save_agent=True &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --dataset_paths="../data/exp_icl//datasets//real/minatar/name=Freeway-MinAtar//dataset.pkl"                                                --exclude_dataset_paths=None                                                                              --save_dir="../data/exp_icl//train_bc/name=Freeway-MinAtar"                                                   --n_iters=2000 --save_agent=True &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --dataset_paths="../data/exp_icl//datasets//real/minatar/name=SpaceInvaders-MinAtar//dataset.pkl"                                          --exclude_dataset_paths=None                                                                              --save_dir="../data/exp_icl//train_bc/name=SpaceInvaders-MinAtar"                                             --n_iters=2000 --save_agent=True &
wait
