#!/bin/bash
source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps/bin/activate
cd /data/vision/phillipi/akumar01/synthetic-mdps/src



CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=dsmdp;i_d=1;i_s=1;t_a=1;t_s=0;o_d=2;tl=64"                         --save_dir="../data/exp_icl//test_bc/name=CartPole-v1/name=dsmdp;i_d=1;i_s=1;t_a=1;t_s=0;o_d=2;tl=64"                                   --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=dsmdp;i_d=4;i_s=3;t_a=3;t_s=2;o_d=4;tl=64"                         --save_dir="../data/exp_icl//test_bc/name=CartPole-v1/name=dsmdp;i_d=4;i_s=3;t_a=3;t_s=2;o_d=4;tl=64"                                   --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=3;t_c=4;o_d=0"                                              --save_dir="../data/exp_icl//test_bc/name=CartPole-v1/name=rf;t_a=3;t_c=4;o_d=0"                                                        --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=1;t_c=3;o_d=0"                                              --save_dir="../data/exp_icl//test_bc/name=CartPole-v1/name=rf;t_a=1;t_c=3;o_d=0"                                                        --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=3;t_c=4;o_d=0"                                              --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/name=rf;t_a=3;t_c=4;o_d=0"                                                         --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=1;t_c=3;o_d=0"                                              --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/name=rf;t_a=1;t_c=3;o_d=0"                                                         --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=0;t_c=1;o_d=4"                                              --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/name=rf;t_a=0;t_c=1;o_d=4"                                                         --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/zero_act"                                                               --save_dir="../data/exp_icl//test_bc/name=Acrobot-v1/zero_act"                                                                          --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=rf;t_a=0;t_c=1;o_d=4"                                              --save_dir="../data/exp_icl//test_bc/name=MountainCar-v0/name=rf;t_a=0;t_c=1;o_d=4"                                                     --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/classic/name=MountainCar-v0//dataset.pkl"        &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/zero_act"                                                               --save_dir="../data/exp_icl//test_bc/name=MountainCar-v0/zero_act"                                                                      --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/classic/name=MountainCar-v0//dataset.pkl"        &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=MountainCar-v0"                                                    --save_dir="../data/exp_icl//test_bc/name=MountainCar-v0/name=MountainCar-v0"                                                           --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/classic/name=MountainCar-v0//dataset.pkl"        &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/all"                                                                    --save_dir="../data/exp_icl//test_bc/name=MountainCar-v0/all"                                                                           --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/classic/name=MountainCar-v0//dataset.pkl"        &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=DiscretePendulum-v1"                                               --save_dir="../data/exp_icl//test_bc/name=DiscretePendulum-v1/name=DiscretePendulum-v1"                                                 --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/classic/name=DiscretePendulum-v1//dataset.pkl"   &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/all"                                                                    --save_dir="../data/exp_icl//test_bc/name=DiscretePendulum-v1/all"                                                                      --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/classic/name=DiscretePendulum-v1//dataset.pkl"   &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/all-name=DiscretePendulum-v1"                                           --save_dir="../data/exp_icl//test_bc/name=DiscretePendulum-v1/n-1"                                                                      --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/classic/name=DiscretePendulum-v1//dataset.pkl"   &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir=None                                                                                               --save_dir="../data/exp_icl//test_bc/name=DiscretePendulum-v1/scratch"                                                                  --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/classic/name=DiscretePendulum-v1//dataset.pkl"   &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/all-name=Asterix-MinAtar"                                               --save_dir="../data/exp_icl//test_bc/name=Asterix-MinAtar/n-1"                                                                          --save_agent=True --n_ckpts=5 --n_iters=500 --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir=None                                                                                               --save_dir="../data/exp_icl//test_bc/name=Asterix-MinAtar/scratch"                                                                      --save_agent=True --n_ckpts=5 --n_iters=500 --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64" --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64"      --save_agent=True --n_ckpts=5 --n_iters=500 --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64"      --save_agent=True --n_ckpts=5 --n_iters=500 --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64" --save_dir="../data/exp_icl//test_bc/name=Freeway-MinAtar/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64"       --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/minatar/name=Freeway-MinAtar//dataset.pkl"       &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=Freeway-MinAtar/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64"       --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/minatar/name=Freeway-MinAtar//dataset.pkl"       &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=Freeway-MinAtar/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64"       --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/minatar/name=Freeway-MinAtar//dataset.pkl"       &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64"                         --save_dir="../data/exp_icl//test_bc/name=Freeway-MinAtar/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64"                               --save_agent=True --n_ckpts=5 --n_iters=100 --dataset_path="../data/exp_icl//datasets//real/minatar/name=Freeway-MinAtar//dataset.pkl"       &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64" --save_dir="../data/exp_icl//test_bc/name=SpaceInvaders-MinAtar/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64" --save_agent=True --n_ckpts=5 --n_iters=500 --dataset_path="../data/exp_icl//datasets//real/minatar/name=SpaceInvaders-MinAtar//dataset.pkl" &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64"                         --save_dir="../data/exp_icl//test_bc/name=SpaceInvaders-MinAtar/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64"                         --save_agent=True --n_ckpts=5 --n_iters=500 --dataset_path="../data/exp_icl//datasets//real/minatar/name=SpaceInvaders-MinAtar//dataset.pkl" &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=dsmdp;i_d=1;i_s=1;t_a=1;t_s=0;o_d=2;tl=64"                         --save_dir="../data/exp_icl//test_bc/name=SpaceInvaders-MinAtar/name=dsmdp;i_d=1;i_s=1;t_a=1;t_s=0;o_d=2;tl=64"                         --save_agent=True --n_ckpts=5 --n_iters=500 --dataset_path="../data/exp_icl//datasets//real/minatar/name=SpaceInvaders-MinAtar//dataset.pkl" &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --load_dir="../data/exp_icl//train_bc/name=dsmdp;i_d=4;i_s=3;t_a=3;t_s=2;o_d=4;tl=64"                         --save_dir="../data/exp_icl//test_bc/name=SpaceInvaders-MinAtar/name=dsmdp;i_d=4;i_s=3;t_a=3;t_s=2;o_d=4;tl=64"                         --save_agent=True --n_ckpts=5 --n_iters=500 --dataset_path="../data/exp_icl//datasets//real/minatar/name=SpaceInvaders-MinAtar//dataset.pkl" &
wait
