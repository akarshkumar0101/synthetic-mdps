#!/bin/bash
source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps/bin/activate
cd /data/vision/phillipi/akumar01/synthetic-mdps/src



CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=3;t_c=4;o_d=0"                                              --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/name=rf;t_a=3;t_c=4;o_d=0_lr=0.0001_pd=1.0"                                                         --n_iters=1000 --lr=0.0001 --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=1;t_c=3;o_d=0"                                              --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/name=rf;t_a=1;t_c=3;o_d=0_lr=0.0001_pd=1.0"                                                         --n_iters=1000 --lr=0.0001 --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=0;t_c=1;o_d=4"                                              --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/name=rf;t_a=0;t_c=1;o_d=4_lr=0.0001_pd=1.0"                                                         --n_iters=1000 --lr=0.0001 --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/zero_act"                                                               --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/zero_act_lr=0.0001_pd=1.0"                                                                          --n_iters=1000 --lr=0.0001 --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --percent_data=1.0 &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=3;t_c=4;o_d=0"                                              --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=rf;t_a=3;t_c=4;o_d=0_lr=0.0001_pd=1.0"                                                   --n_iters=1000 --lr=0.0001 --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=1;t_c=3;o_d=0"                                              --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=rf;t_a=1;t_c=3;o_d=0_lr=0.0001_pd=1.0"                                                   --n_iters=1000 --lr=0.0001 --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=0;t_c=1;o_d=4"                                              --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=rf;t_a=0;t_c=1;o_d=4_lr=0.0001_pd=1.0"                                                   --n_iters=1000 --lr=0.0001 --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/zero_act"                                                               --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/zero_act_lr=0.0001_pd=1.0"                                                                    --n_iters=1000 --lr=0.0001 --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --percent_data=1.0 &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=3;t_c=4;o_d=0"                                              --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/name=rf;t_a=3;t_c=4;o_d=0_lr=0.0003_pd=1.0"                                                         --n_iters=1000 --lr=0.0003 --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=1;t_c=3;o_d=0"                                              --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/name=rf;t_a=1;t_c=3;o_d=0_lr=0.0003_pd=1.0"                                                         --n_iters=1000 --lr=0.0003 --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=0;t_c=1;o_d=4"                                              --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/name=rf;t_a=0;t_c=1;o_d=4_lr=0.0003_pd=1.0"                                                         --n_iters=1000 --lr=0.0003 --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/zero_act"                                                               --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/zero_act_lr=0.0003_pd=1.0"                                                                          --n_iters=1000 --lr=0.0003 --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --percent_data=1.0 &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=3;t_c=4;o_d=0"                                              --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=rf;t_a=3;t_c=4;o_d=0_lr=0.0003_pd=1.0"                                                   --n_iters=1000 --lr=0.0003 --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=1;t_c=3;o_d=0"                                              --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=rf;t_a=1;t_c=3;o_d=0_lr=0.0003_pd=1.0"                                                   --n_iters=1000 --lr=0.0003 --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=0;t_c=1;o_d=4"                                              --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=rf;t_a=0;t_c=1;o_d=4_lr=0.0003_pd=1.0"                                                   --n_iters=1000 --lr=0.0003 --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/zero_act"                                                               --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/zero_act_lr=0.0003_pd=1.0"                                                                    --n_iters=1000 --lr=0.0003 --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --percent_data=1.0 &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=3;t_c=4;o_d=0"                                              --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/name=rf;t_a=3;t_c=4;o_d=0_lr=0.001_pd=1.0"                                                          --n_iters=1000 --lr=0.001  --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=1;t_c=3;o_d=0"                                              --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/name=rf;t_a=1;t_c=3;o_d=0_lr=0.001_pd=1.0"                                                          --n_iters=1000 --lr=0.001  --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=0;t_c=1;o_d=4"                                              --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/name=rf;t_a=0;t_c=1;o_d=4_lr=0.001_pd=1.0"                                                          --n_iters=1000 --lr=0.001  --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/zero_act"                                                               --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/zero_act_lr=0.001_pd=1.0"                                                                           --n_iters=1000 --lr=0.001  --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --percent_data=1.0 &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=3;t_c=4;o_d=0"                                              --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=rf;t_a=3;t_c=4;o_d=0_lr=0.001_pd=1.0"                                                    --n_iters=1000 --lr=0.001  --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=1;t_c=3;o_d=0"                                              --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=rf;t_a=1;t_c=3;o_d=0_lr=0.001_pd=1.0"                                                    --n_iters=1000 --lr=0.001  --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=0;t_c=1;o_d=4"                                              --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=rf;t_a=0;t_c=1;o_d=4_lr=0.001_pd=1.0"                                                    --n_iters=1000 --lr=0.001  --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/zero_act"                                                               --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/zero_act_lr=0.001_pd=1.0"                                                                     --n_iters=1000 --lr=0.001  --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --percent_data=1.0 &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=3;t_c=4;o_d=0"                                              --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/name=rf;t_a=3;t_c=4;o_d=0_lr=0.003_pd=1.0"                                                          --n_iters=1000 --lr=0.003  --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=1;t_c=3;o_d=0"                                              --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/name=rf;t_a=1;t_c=3;o_d=0_lr=0.003_pd=1.0"                                                          --n_iters=1000 --lr=0.003  --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=0;t_c=1;o_d=4"                                              --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/name=rf;t_a=0;t_c=1;o_d=4_lr=0.003_pd=1.0"                                                          --n_iters=1000 --lr=0.003  --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/zero_act"                                                               --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/zero_act_lr=0.003_pd=1.0"                                                                           --n_iters=1000 --lr=0.003  --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --percent_data=1.0 &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=3;t_c=4;o_d=0"                                              --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=rf;t_a=3;t_c=4;o_d=0_lr=0.003_pd=1.0"                                                    --n_iters=1000 --lr=0.003  --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=1;t_c=3;o_d=0"                                              --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=rf;t_a=1;t_c=3;o_d=0_lr=0.003_pd=1.0"                                                    --n_iters=1000 --lr=0.003  --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=0;t_c=1;o_d=4"                                              --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=rf;t_a=0;t_c=1;o_d=4_lr=0.003_pd=1.0"                                                    --n_iters=1000 --lr=0.003  --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/zero_act"                                                               --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/zero_act_lr=0.003_pd=1.0"                                                                     --n_iters=1000 --lr=0.003  --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --percent_data=1.0 &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=3;t_c=4;o_d=0"                                              --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/name=rf;t_a=3;t_c=4;o_d=0_lr=0.01_pd=1.0"                                                           --n_iters=1000 --lr=0.01   --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=1;t_c=3;o_d=0"                                              --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/name=rf;t_a=1;t_c=3;o_d=0_lr=0.01_pd=1.0"                                                           --n_iters=1000 --lr=0.01   --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=0;t_c=1;o_d=4"                                              --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/name=rf;t_a=0;t_c=1;o_d=4_lr=0.01_pd=1.0"                                                           --n_iters=1000 --lr=0.01   --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/zero_act"                                                               --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/zero_act_lr=0.01_pd=1.0"                                                                            --n_iters=1000 --lr=0.01   --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --percent_data=1.0 &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=3;t_c=4;o_d=0"                                              --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=rf;t_a=3;t_c=4;o_d=0_lr=0.01_pd=1.0"                                                     --n_iters=1000 --lr=0.01   --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=1;t_c=3;o_d=0"                                              --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=rf;t_a=1;t_c=3;o_d=0_lr=0.01_pd=1.0"                                                     --n_iters=1000 --lr=0.01   --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=0;t_c=1;o_d=4"                                              --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=rf;t_a=0;t_c=1;o_d=4_lr=0.01_pd=1.0"                                                     --n_iters=1000 --lr=0.01   --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/zero_act"                                                               --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/zero_act_lr=0.01_pd=1.0"                                                                      --n_iters=1000 --lr=0.01   --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --percent_data=1.0 &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=3;t_c=4;o_d=0"                                              --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/name=rf;t_a=3;t_c=4;o_d=0_lr=0.03_pd=1.0"                                                           --n_iters=1000 --lr=0.03   --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=1;t_c=3;o_d=0"                                              --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/name=rf;t_a=1;t_c=3;o_d=0_lr=0.03_pd=1.0"                                                           --n_iters=1000 --lr=0.03   --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=0;t_c=1;o_d=4"                                              --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/name=rf;t_a=0;t_c=1;o_d=4_lr=0.03_pd=1.0"                                                           --n_iters=1000 --lr=0.03   --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/zero_act"                                                               --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/zero_act_lr=0.03_pd=1.0"                                                                            --n_iters=1000 --lr=0.03   --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --percent_data=1.0 &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=3;t_c=4;o_d=0"                                              --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=rf;t_a=3;t_c=4;o_d=0_lr=0.03_pd=1.0"                                                     --n_iters=1000 --lr=0.03   --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=1;t_c=3;o_d=0"                                              --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=rf;t_a=1;t_c=3;o_d=0_lr=0.03_pd=1.0"                                                     --n_iters=1000 --lr=0.03   --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=0;t_c=1;o_d=4"                                              --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=rf;t_a=0;t_c=1;o_d=4_lr=0.03_pd=1.0"                                                     --n_iters=1000 --lr=0.03   --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/zero_act"                                                               --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/zero_act_lr=0.03_pd=1.0"                                                                      --n_iters=1000 --lr=0.03   --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --percent_data=1.0 &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=3;t_c=4;o_d=0"                                              --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/name=rf;t_a=3;t_c=4;o_d=0_lr=0.1_pd=1.0"                                                            --n_iters=1000 --lr=0.1    --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=1;t_c=3;o_d=0"                                              --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/name=rf;t_a=1;t_c=3;o_d=0_lr=0.1_pd=1.0"                                                            --n_iters=1000 --lr=0.1    --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=0;t_c=1;o_d=4"                                              --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/name=rf;t_a=0;t_c=1;o_d=4_lr=0.1_pd=1.0"                                                            --n_iters=1000 --lr=0.1    --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/zero_act"                                                               --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/zero_act_lr=0.1_pd=1.0"                                                                             --n_iters=1000 --lr=0.1    --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --percent_data=1.0 &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=3;t_c=4;o_d=0"                                              --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=rf;t_a=3;t_c=4;o_d=0_lr=0.1_pd=1.0"                                                      --n_iters=1000 --lr=0.1    --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=1;t_c=3;o_d=0"                                              --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=rf;t_a=1;t_c=3;o_d=0_lr=0.1_pd=1.0"                                                      --n_iters=1000 --lr=0.1    --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=0;t_c=1;o_d=4"                                              --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=rf;t_a=0;t_c=1;o_d=4_lr=0.1_pd=1.0"                                                      --n_iters=1000 --lr=0.1    --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/zero_act"                                                               --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/zero_act_lr=0.1_pd=1.0"                                                                       --n_iters=1000 --lr=0.1    --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --percent_data=1.0 &
wait
