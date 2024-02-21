#!/bin/bash
source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps/bin/activate
cd /data/vision/phillipi/akumar01/synthetic-mdps/src



CUDA_VISIBLE_DEVICES=0 python icl_gen.py --env_id="name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64" --save_dir="../data/exp_icl//datasets//synthetic/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64//" --n_seeds_seq=16 --n_seeds_par=16 --n_iters_train=100 --lr=0.0003 &
CUDA_VISIBLE_DEVICES=1 python icl_gen.py --env_id="name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64" --save_dir="../data/exp_icl//datasets//synthetic/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64//" --n_seeds_seq=16 --n_seeds_par=16 --n_iters_train=100 --lr=0.0003 &
CUDA_VISIBLE_DEVICES=2 python icl_gen.py --env_id="name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64" --save_dir="../data/exp_icl//datasets//synthetic/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64//" --n_seeds_seq=16 --n_seeds_par=16 --n_iters_train=100 --lr=0.0003 &
CUDA_VISIBLE_DEVICES=3 python icl_gen.py --env_id="name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64"                         --save_dir="../data/exp_icl//datasets//synthetic/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64//"                         --n_seeds_seq=16 --n_seeds_par=16 --n_iters_train=100 --lr=0.0003 &
wait
