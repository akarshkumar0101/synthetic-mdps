#!/bin/bash
source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps/bin/activate
cd /data/vision/phillipi/akumar01/synthetic-mdps/src



CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64" --save_dir="../data/exp_icl//test_bc/name=CartPole-v1/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64_lr=0.0001_pd=1.0"           --n_iters=1000 --lr=0.0001 --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=CartPole-v1/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64_lr=0.0001_pd=1.0"           --n_iters=1000 --lr=0.0001 --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=CartPole-v1/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64_lr=0.0001_pd=1.0"           --n_iters=1000 --lr=0.0001 --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64"                         --save_dir="../data/exp_icl//test_bc/name=CartPole-v1/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64_lr=0.0001_pd=1.0"                                   --n_iters=1000 --lr=0.0001 --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --percent_data=1.0 &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64" --save_dir="../data/exp_icl//test_bc/name=Asterix-MinAtar/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64_lr=0.0001_pd=1.0"       --n_iters=1000 --lr=0.0001 --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=Asterix-MinAtar/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64_lr=0.0001_pd=1.0"       --n_iters=1000 --lr=0.0001 --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=Asterix-MinAtar/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64_lr=0.0001_pd=1.0"       --n_iters=1000 --lr=0.0001 --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64"                         --save_dir="../data/exp_icl//test_bc/name=Asterix-MinAtar/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64_lr=0.0001_pd=1.0"                               --n_iters=1000 --lr=0.0001 --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --percent_data=1.0 &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64" --save_dir="../data/exp_icl//test_bc/name=CartPole-v1/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64_lr=0.0003_pd=1.0"           --n_iters=1000 --lr=0.0003 --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=CartPole-v1/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64_lr=0.0003_pd=1.0"           --n_iters=1000 --lr=0.0003 --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=CartPole-v1/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64_lr=0.0003_pd=1.0"           --n_iters=1000 --lr=0.0003 --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64"                         --save_dir="../data/exp_icl//test_bc/name=CartPole-v1/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64_lr=0.0003_pd=1.0"                                   --n_iters=1000 --lr=0.0003 --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --percent_data=1.0 &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64" --save_dir="../data/exp_icl//test_bc/name=Asterix-MinAtar/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64_lr=0.0003_pd=1.0"       --n_iters=1000 --lr=0.0003 --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=Asterix-MinAtar/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64_lr=0.0003_pd=1.0"       --n_iters=1000 --lr=0.0003 --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=Asterix-MinAtar/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64_lr=0.0003_pd=1.0"       --n_iters=1000 --lr=0.0003 --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64"                         --save_dir="../data/exp_icl//test_bc/name=Asterix-MinAtar/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64_lr=0.0003_pd=1.0"                               --n_iters=1000 --lr=0.0003 --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --percent_data=1.0 &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64" --save_dir="../data/exp_icl//test_bc/name=CartPole-v1/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64_lr=0.001_pd=1.0"            --n_iters=1000 --lr=0.001  --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=CartPole-v1/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64_lr=0.001_pd=1.0"            --n_iters=1000 --lr=0.001  --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=CartPole-v1/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64_lr=0.001_pd=1.0"            --n_iters=1000 --lr=0.001  --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64"                         --save_dir="../data/exp_icl//test_bc/name=CartPole-v1/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64_lr=0.001_pd=1.0"                                    --n_iters=1000 --lr=0.001  --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --percent_data=1.0 &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64" --save_dir="../data/exp_icl//test_bc/name=Asterix-MinAtar/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64_lr=0.001_pd=1.0"        --n_iters=1000 --lr=0.001  --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=Asterix-MinAtar/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64_lr=0.001_pd=1.0"        --n_iters=1000 --lr=0.001  --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=Asterix-MinAtar/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64_lr=0.001_pd=1.0"        --n_iters=1000 --lr=0.001  --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64"                         --save_dir="../data/exp_icl//test_bc/name=Asterix-MinAtar/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64_lr=0.001_pd=1.0"                                --n_iters=1000 --lr=0.001  --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --percent_data=1.0 &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64" --save_dir="../data/exp_icl//test_bc/name=CartPole-v1/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64_lr=0.003_pd=1.0"            --n_iters=1000 --lr=0.003  --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=CartPole-v1/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64_lr=0.003_pd=1.0"            --n_iters=1000 --lr=0.003  --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=CartPole-v1/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64_lr=0.003_pd=1.0"            --n_iters=1000 --lr=0.003  --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64"                         --save_dir="../data/exp_icl//test_bc/name=CartPole-v1/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64_lr=0.003_pd=1.0"                                    --n_iters=1000 --lr=0.003  --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --percent_data=1.0 &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64" --save_dir="../data/exp_icl//test_bc/name=Asterix-MinAtar/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64_lr=0.003_pd=1.0"        --n_iters=1000 --lr=0.003  --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=Asterix-MinAtar/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64_lr=0.003_pd=1.0"        --n_iters=1000 --lr=0.003  --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=Asterix-MinAtar/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64_lr=0.003_pd=1.0"        --n_iters=1000 --lr=0.003  --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64"                         --save_dir="../data/exp_icl//test_bc/name=Asterix-MinAtar/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64_lr=0.003_pd=1.0"                                --n_iters=1000 --lr=0.003  --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --percent_data=1.0 &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64" --save_dir="../data/exp_icl//test_bc/name=CartPole-v1/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64_lr=0.01_pd=1.0"             --n_iters=1000 --lr=0.01   --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=CartPole-v1/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64_lr=0.01_pd=1.0"             --n_iters=1000 --lr=0.01   --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=CartPole-v1/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64_lr=0.01_pd=1.0"             --n_iters=1000 --lr=0.01   --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64"                         --save_dir="../data/exp_icl//test_bc/name=CartPole-v1/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64_lr=0.01_pd=1.0"                                     --n_iters=1000 --lr=0.01   --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --percent_data=1.0 &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64" --save_dir="../data/exp_icl//test_bc/name=Asterix-MinAtar/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64_lr=0.01_pd=1.0"         --n_iters=1000 --lr=0.01   --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=Asterix-MinAtar/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64_lr=0.01_pd=1.0"         --n_iters=1000 --lr=0.01   --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=Asterix-MinAtar/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64_lr=0.01_pd=1.0"         --n_iters=1000 --lr=0.01   --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64"                         --save_dir="../data/exp_icl//test_bc/name=Asterix-MinAtar/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64_lr=0.01_pd=1.0"                                 --n_iters=1000 --lr=0.01   --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --percent_data=1.0 &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64" --save_dir="../data/exp_icl//test_bc/name=CartPole-v1/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64_lr=0.03_pd=1.0"             --n_iters=1000 --lr=0.03   --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=CartPole-v1/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64_lr=0.03_pd=1.0"             --n_iters=1000 --lr=0.03   --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=CartPole-v1/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64_lr=0.03_pd=1.0"             --n_iters=1000 --lr=0.03   --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64"                         --save_dir="../data/exp_icl//test_bc/name=CartPole-v1/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64_lr=0.03_pd=1.0"                                     --n_iters=1000 --lr=0.03   --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --percent_data=1.0 &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64" --save_dir="../data/exp_icl//test_bc/name=Asterix-MinAtar/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64_lr=0.03_pd=1.0"         --n_iters=1000 --lr=0.03   --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=Asterix-MinAtar/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64_lr=0.03_pd=1.0"         --n_iters=1000 --lr=0.03   --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=Asterix-MinAtar/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64_lr=0.03_pd=1.0"         --n_iters=1000 --lr=0.03   --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64"                         --save_dir="../data/exp_icl//test_bc/name=Asterix-MinAtar/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64_lr=0.03_pd=1.0"                                 --n_iters=1000 --lr=0.03   --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --percent_data=1.0 &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64" --save_dir="../data/exp_icl//test_bc/name=CartPole-v1/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64_lr=0.1_pd=1.0"              --n_iters=1000 --lr=0.1    --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=CartPole-v1/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64_lr=0.1_pd=1.0"              --n_iters=1000 --lr=0.1    --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=CartPole-v1/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64_lr=0.1_pd=1.0"              --n_iters=1000 --lr=0.1    --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64"                         --save_dir="../data/exp_icl//test_bc/name=CartPole-v1/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64_lr=0.1_pd=1.0"                                      --n_iters=1000 --lr=0.1    --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --percent_data=1.0 &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64" --save_dir="../data/exp_icl//test_bc/name=Asterix-MinAtar/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64_lr=0.1_pd=1.0"          --n_iters=1000 --lr=0.1    --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=Asterix-MinAtar/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64_lr=0.1_pd=1.0"          --n_iters=1000 --lr=0.1    --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=Asterix-MinAtar/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64_lr=0.1_pd=1.0"          --n_iters=1000 --lr=0.1    --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --percent_data=1.0 &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64"                         --save_dir="../data/exp_icl//test_bc/name=Asterix-MinAtar/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64_lr=0.1_pd=1.0"                                  --n_iters=1000 --lr=0.1    --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --percent_data=1.0 &
wait
