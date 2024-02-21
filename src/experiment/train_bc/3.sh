#!/bin/bash
source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps/bin/activate
cd /data/vision/phillipi/akumar01/synthetic-mdps/src



CUDA_VISIBLE_DEVICES=0 python icl_bc.py --dataset_paths="../data/exp_icl//datasets//real/classic/name=MountainCar-v0//dataset.pkl"                                                 --exclude_dataset_paths=None                                                                              --save_dir="../data/exp_icl//train_bc/name=MountainCar-v0"                                                    --save_agent=True --n_iters=3000 &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --dataset_paths="../data/exp_icl//datasets//real/classic/name=DiscretePendulum-v1//dataset.pkl"                                            --exclude_dataset_paths=None                                                                              --save_dir="../data/exp_icl//train_bc/name=DiscretePendulum-v1"                                               --save_agent=True --n_iters=3000 &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --dataset_paths="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"                                                --exclude_dataset_paths=None                                                                              --save_dir="../data/exp_icl//train_bc/name=Asterix-MinAtar"                                                   --save_agent=True --n_iters=3000 &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --dataset_paths="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"                                               --exclude_dataset_paths=None                                                                              --save_dir="../data/exp_icl//train_bc/name=Breakout-MinAtar"                                                  --save_agent=True --n_iters=3000 &
wait
