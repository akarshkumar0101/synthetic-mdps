#!/bin/bash
source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps/bin/activate
cd /data/vision/phillipi/akumar01/synthetic-mdps/src



CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64" --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64"            --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64"            --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/random_function"                                                        --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/random_function"                                                                   --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/zero_act"                                                               --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/zero_act"                                                                          --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64" --save_dir="../data/exp_icl//test_bc/name=DiscretePendulum-v1/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64"   --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/classic/name=DiscretePendulum-v1//dataset.pkl"   &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=DiscretePendulum-v1/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64"   --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/classic/name=DiscretePendulum-v1//dataset.pkl"   &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/random_function"                                                        --save_dir="../data/exp_icl//test_bc/name=DiscretePendulum-v1/random_function"                                                          --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/classic/name=DiscretePendulum-v1//dataset.pkl"   &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/zero_act"                                                               --save_dir="../data/exp_icl//test_bc/name=DiscretePendulum-v1/zero_act"                                                                 --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/classic/name=DiscretePendulum-v1//dataset.pkl"   &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64" --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64"      --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64"      --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/random_function"                                                        --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/random_function"                                                             --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/zero_act"                                                               --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/zero_act"                                                                    --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64" --save_dir="../data/exp_icl//test_bc/name=SpaceInvaders-MinAtar/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64" --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/minatar/name=SpaceInvaders-MinAtar//dataset.pkl" &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=SpaceInvaders-MinAtar/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64" --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/minatar/name=SpaceInvaders-MinAtar//dataset.pkl" &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/random_function"                                                        --save_dir="../data/exp_icl//test_bc/name=SpaceInvaders-MinAtar/random_function"                                                        --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/minatar/name=SpaceInvaders-MinAtar//dataset.pkl" &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/zero_act"                                                               --save_dir="../data/exp_icl//test_bc/name=SpaceInvaders-MinAtar/zero_act"                                                               --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/minatar/name=SpaceInvaders-MinAtar//dataset.pkl" &
wait
