#!/bin/bash
source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps/bin/activate
cd /data/vision/phillipi/akumar01/synthetic-mdps/src



CUDA_VISIBLE_DEVICES=0 python icl_bc.py --dataset_paths="../data/exp_icl//datasets//synthetic/name=dsmdp;i_d=0;i_s=3;t_a=1;t_s=1;o_d=3;tl=64//dataset.pkl"                         --exclude_dataset_paths=None                                                                              --save_dir="../data/exp_icl//train_bc/name=dsmdp;i_d=0;i_s=3;t_a=1;t_s=1;o_d=3;tl=64"                         --save_agent=True --n_iters_eval=2000 --n_iters=250000 --n_augs=1000000000 &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --dataset_paths="../data/exp_icl//datasets//synthetic/name=dsmdp;i_d=0;i_s=3;t_a=1;t_s=1;o_d=3;tl=1//dataset.pkl"                          --exclude_dataset_paths=None                                                                              --save_dir="../data/exp_icl//train_bc/name=dsmdp;i_d=0;i_s=3;t_a=1;t_s=1;o_d=3;tl=1"                          --save_agent=True --n_iters_eval=2000 --n_iters=250000 --n_augs=1000000000 &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --dataset_paths="../data/exp_icl//datasets//synthetic/name=dsmdp;i_d=4;i_s=0;t_a=1;t_s=3;o_d=4;tl=64//dataset.pkl"                         --exclude_dataset_paths=None                                                                              --save_dir="../data/exp_icl//train_bc/name=dsmdp;i_d=4;i_s=0;t_a=1;t_s=3;o_d=4;tl=64"                         --save_agent=True --n_iters_eval=2000 --n_iters=250000 --n_augs=1000000000 &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --dataset_paths="../data/exp_icl//datasets//synthetic/name=dsmdp;i_d=4;i_s=0;t_a=1;t_s=3;o_d=4;tl=1//dataset.pkl"                          --exclude_dataset_paths=None                                                                              --save_dir="../data/exp_icl//train_bc/name=dsmdp;i_d=4;i_s=0;t_a=1;t_s=3;o_d=4;tl=1"                          --save_agent=True --n_iters_eval=2000 --n_iters=250000 --n_augs=1000000000 &
CUDA_VISIBLE_DEVICES=4 python icl_bc.py --dataset_paths="../data/exp_icl//datasets//synthetic/name=dsmdp;i_d=2;i_s=4;t_a=0;t_s=3;o_d=1;tl=64//dataset.pkl"                         --exclude_dataset_paths=None                                                                              --save_dir="../data/exp_icl//train_bc/name=dsmdp;i_d=2;i_s=4;t_a=0;t_s=3;o_d=1;tl=64"                         --save_agent=True --n_iters_eval=2000 --n_iters=250000 --n_augs=1000000000 &
CUDA_VISIBLE_DEVICES=5 python icl_bc.py --dataset_paths="../data/exp_icl//datasets//synthetic/name=dsmdp;i_d=2;i_s=4;t_a=0;t_s=3;o_d=1;tl=1//dataset.pkl"                          --exclude_dataset_paths=None                                                                              --save_dir="../data/exp_icl//train_bc/name=dsmdp;i_d=2;i_s=4;t_a=0;t_s=3;o_d=1;tl=1"                          --save_agent=True --n_iters_eval=2000 --n_iters=250000 --n_augs=1000000000 &
CUDA_VISIBLE_DEVICES=6 python icl_bc.py --dataset_paths="../data/exp_icl//datasets//synthetic/name=dsmdp;i_d=2;i_s=0;t_a=4;t_s=1;o_d=2;tl=64//dataset.pkl"                         --exclude_dataset_paths=None                                                                              --save_dir="../data/exp_icl//train_bc/name=dsmdp;i_d=2;i_s=0;t_a=4;t_s=1;o_d=2;tl=64"                         --save_agent=True --n_iters_eval=2000 --n_iters=250000 --n_augs=1000000000 &
CUDA_VISIBLE_DEVICES=7 python icl_bc.py --dataset_paths="../data/exp_icl//datasets//synthetic/name=dsmdp;i_d=2;i_s=0;t_a=4;t_s=1;o_d=2;tl=1//dataset.pkl"                          --exclude_dataset_paths=None                                                                              --save_dir="../data/exp_icl//train_bc/name=dsmdp;i_d=2;i_s=0;t_a=4;t_s=1;o_d=2;tl=1"                          --save_agent=True --n_iters_eval=2000 --n_iters=250000 --n_augs=1000000000 &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --dataset_paths="../data/exp_icl//datasets/real/*/*/dataset.pkl"                                                                           --exclude_dataset_paths=None                                                                              --save_dir="../data/exp_icl//train_bc/all"                                                                    --save_agent=True --n_iters_eval=2000 --n_iters=250000 --n_augs=1000000000 &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --dataset_paths="../data/exp_icl//datasets/real/*/*/dataset.pkl"                                                                           --exclude_dataset_paths="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --save_dir="../data/exp_icl//train_bc/all-name=CartPole-v1"                                                   --save_agent=True --n_iters_eval=2000 --n_iters=250000 --n_augs=1000000000 &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --dataset_paths="../data/exp_icl//datasets/real/*/*/dataset.pkl"                                                                           --exclude_dataset_paths="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --save_dir="../data/exp_icl//train_bc/all-name=Acrobot-v1"                                                    --save_agent=True --n_iters_eval=2000 --n_iters=250000 --n_augs=1000000000 &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --dataset_paths="../data/exp_icl//datasets/real/*/*/dataset.pkl"                                                                           --exclude_dataset_paths="../data/exp_icl//datasets//real/classic/name=MountainCar-v0//dataset.pkl"        --save_dir="../data/exp_icl//train_bc/all-name=MountainCar-v0"                                                --save_agent=True --n_iters_eval=2000 --n_iters=250000 --n_augs=1000000000 &
CUDA_VISIBLE_DEVICES=4 python icl_bc.py --dataset_paths="../data/exp_icl//datasets/real/*/*/dataset.pkl"                                                                           --exclude_dataset_paths="../data/exp_icl//datasets//real/classic/name=DiscretePendulum-v1//dataset.pkl"   --save_dir="../data/exp_icl//train_bc/all-name=DiscretePendulum-v1"                                           --save_agent=True --n_iters_eval=2000 --n_iters=250000 --n_augs=1000000000 &
CUDA_VISIBLE_DEVICES=5 python icl_bc.py --dataset_paths="../data/exp_icl//datasets/real/*/*/dataset.pkl"                                                                           --exclude_dataset_paths="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --save_dir="../data/exp_icl//train_bc/all-name=Asterix-MinAtar"                                               --save_agent=True --n_iters_eval=2000 --n_iters=250000 --n_augs=1000000000 &
CUDA_VISIBLE_DEVICES=6 python icl_bc.py --dataset_paths="../data/exp_icl//datasets/real/*/*/dataset.pkl"                                                                           --exclude_dataset_paths="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --save_dir="../data/exp_icl//train_bc/all-name=Breakout-MinAtar"                                              --save_agent=True --n_iters_eval=2000 --n_iters=250000 --n_augs=1000000000 &
CUDA_VISIBLE_DEVICES=7 python icl_bc.py --dataset_paths="../data/exp_icl//datasets/real/*/*/dataset.pkl"                                                                           --exclude_dataset_paths="../data/exp_icl//datasets//real/minatar/name=Freeway-MinAtar//dataset.pkl"       --save_dir="../data/exp_icl//train_bc/all-name=Freeway-MinAtar"                                               --save_agent=True --n_iters_eval=2000 --n_iters=250000 --n_augs=1000000000 &
wait
