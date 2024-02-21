#!/bin/bash
source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps/bin/activate
cd /data/vision/phillipi/akumar01/synthetic-mdps/src



CUDA_VISIBLE_DEVICES=0 python icl_gen.py --env_id="name=rf;t_a=0;t_c=1;o_d=4"                                              --save_dir="../data/exp_icl//datasets//synthetic/name=rf;t_a=0;t_c=1;o_d=4//"                                              --n_seeds_seq=16 --n_seeds_par=16 --n_iters_train=100 --lr=0.0003 &
CUDA_VISIBLE_DEVICES=1 python icl_gen.py --env_id="zero_act"                                                               --save_dir="../data/exp_icl//datasets//synthetic/zero_act//"                                                               --n_seeds_seq=16 --n_seeds_par=16 --n_iters_train=100 --lr=0.0003 &
wait
