#!/bin/bash
source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps/bin/activate
cd /data/vision/phillipi/akumar01/synthetic-mdps/src



CUDA_VISIBLE_DEVICES=0 python icl_bc.py --name="pretrain_name=csmdp;i_d=2;i_s=3;t_a=4;t_c=1;t_l=2;t_s=3;o_d=1;o_c=2;r_c=1;tl=64" --dataset_path="../data/exp_icl//datasets/name=csmdp;i_d=2;i_s=3;t_a=4;t_c=1;t_l=2;t_s=3;o_d=1;o_c=2;r_c=1;tl=64/dataset.pkl" --save_dir="../data/exp_icl//train_wm/name=csmdp;i_d=2;i_s=3;t_a=4;t_c=1;t_l=2;t_s=3;o_d=1;o_c=2;r_c=1;tl=64" --save_agent=True --n_iters=2000 --obj="wm" &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --name="pretrain_name=csmdp;i_d=4;i_s=2;t_a=3;t_c=0;t_l=3;t_s=2;o_d=3;o_c=0;r_c=0;tl=64" --dataset_path="../data/exp_icl//datasets/name=csmdp;i_d=4;i_s=2;t_a=3;t_c=0;t_l=3;t_s=2;o_d=3;o_c=0;r_c=0;tl=64/dataset.pkl" --save_dir="../data/exp_icl//train_wm/name=csmdp;i_d=4;i_s=2;t_a=3;t_c=0;t_l=3;t_s=2;o_d=3;o_c=0;r_c=0;tl=64" --save_agent=True --n_iters=2000 --obj="wm" &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --name="pretrain_name=csmdp;i_d=0;i_s=3;t_a=2;t_c=3;t_l=0;t_s=4;o_d=0;o_c=0;r_c=2;tl=64" --dataset_path="../data/exp_icl//datasets/name=csmdp;i_d=0;i_s=3;t_a=2;t_c=3;t_l=0;t_s=4;o_d=0;o_c=0;r_c=2;tl=64/dataset.pkl" --save_dir="../data/exp_icl//train_wm/name=csmdp;i_d=0;i_s=3;t_a=2;t_c=3;t_l=0;t_s=4;o_d=0;o_c=0;r_c=2;tl=64" --save_agent=True --n_iters=2000 --obj="wm" &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --name="pretrain_name=csmdp;i_d=3;i_s=2;t_a=3;t_c=0;t_l=0;t_s=0;o_d=3;o_c=0;r_c=2;tl=64" --dataset_path="../data/exp_icl//datasets/name=csmdp;i_d=3;i_s=2;t_a=3;t_c=0;t_l=0;t_s=0;o_d=3;o_c=0;r_c=2;tl=64/dataset.pkl" --save_dir="../data/exp_icl//train_wm/name=csmdp;i_d=3;i_s=2;t_a=3;t_c=0;t_l=0;t_s=0;o_d=3;o_c=0;r_c=2;tl=64" --save_agent=True --n_iters=2000 --obj="wm" &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --name="pretrain_name=csmdp;i_d=2;i_s=3;t_a=3;t_c=1;t_l=2;t_s=4;o_d=3;o_c=3;r_c=1;tl=64" --dataset_path="../data/exp_icl//datasets/name=csmdp;i_d=2;i_s=3;t_a=3;t_c=1;t_l=2;t_s=4;o_d=3;o_c=3;r_c=1;tl=64/dataset.pkl" --save_dir="../data/exp_icl//train_wm/name=csmdp;i_d=2;i_s=3;t_a=3;t_c=1;t_l=2;t_s=4;o_d=3;o_c=3;r_c=1;tl=64" --save_agent=True --n_iters=2000 --obj="wm" &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --name="pretrain_name=csmdp;i_d=1;i_s=1;t_a=1;t_c=2;t_l=0;t_s=3;o_d=1;o_c=4;r_c=3;tl=64" --dataset_path="../data/exp_icl//datasets/name=csmdp;i_d=1;i_s=1;t_a=1;t_c=2;t_l=0;t_s=3;o_d=1;o_c=4;r_c=3;tl=64/dataset.pkl" --save_dir="../data/exp_icl//train_wm/name=csmdp;i_d=1;i_s=1;t_a=1;t_c=2;t_l=0;t_s=3;o_d=1;o_c=4;r_c=3;tl=64" --save_agent=True --n_iters=2000 --obj="wm" &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --name="pretrain_name=csmdp;i_d=1;i_s=1;t_a=2;t_c=0;t_l=0;t_s=2;o_d=4;o_c=0;r_c=4;tl=64" --dataset_path="../data/exp_icl//datasets/name=csmdp;i_d=1;i_s=1;t_a=2;t_c=0;t_l=0;t_s=2;o_d=4;o_c=0;r_c=4;tl=64/dataset.pkl" --save_dir="../data/exp_icl//train_wm/name=csmdp;i_d=1;i_s=1;t_a=2;t_c=0;t_l=0;t_s=2;o_d=4;o_c=0;r_c=4;tl=64" --save_agent=True --n_iters=2000 --obj="wm" &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --name="pretrain_name=csmdp;i_d=3;i_s=3;t_a=0;t_c=0;t_l=3;t_s=3;o_d=0;o_c=3;r_c=0;tl=64" --dataset_path="../data/exp_icl//datasets/name=csmdp;i_d=3;i_s=3;t_a=0;t_c=0;t_l=3;t_s=3;o_d=0;o_c=3;r_c=0;tl=64/dataset.pkl" --save_dir="../data/exp_icl//train_wm/name=csmdp;i_d=3;i_s=3;t_a=0;t_c=0;t_l=3;t_s=3;o_d=0;o_c=3;r_c=0;tl=64" --save_agent=True --n_iters=2000 --obj="wm" &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --name="pretrain_name=csmdp;i_d=3;i_s=3;t_a=3;t_c=4;t_l=2;t_s=1;o_d=2;o_c=2;r_c=1;tl=64" --dataset_path="../data/exp_icl//datasets/name=csmdp;i_d=3;i_s=3;t_a=3;t_c=4;t_l=2;t_s=1;o_d=2;o_c=2;r_c=1;tl=64/dataset.pkl" --save_dir="../data/exp_icl//train_wm/name=csmdp;i_d=3;i_s=3;t_a=3;t_c=4;t_l=2;t_s=1;o_d=2;o_c=2;r_c=1;tl=64" --save_agent=True --n_iters=2000 --obj="wm" &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --name="pretrain_name=csmdp;i_d=3;i_s=3;t_a=4;t_c=4;t_l=2;t_s=1;o_d=1;o_c=0;r_c=0;tl=64" --dataset_path="../data/exp_icl//datasets/name=csmdp;i_d=3;i_s=3;t_a=4;t_c=4;t_l=2;t_s=1;o_d=1;o_c=0;r_c=0;tl=64/dataset.pkl" --save_dir="../data/exp_icl//train_wm/name=csmdp;i_d=3;i_s=3;t_a=4;t_c=4;t_l=2;t_s=1;o_d=1;o_c=0;r_c=0;tl=64" --save_agent=True --n_iters=2000 --obj="wm" &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --name="pretrain_name=csmdp;i_d=2;i_s=4;t_a=3;t_c=3;t_l=2;t_s=1;o_d=0;o_c=4;r_c=1;tl=64" --dataset_path="../data/exp_icl//datasets/name=csmdp;i_d=2;i_s=4;t_a=3;t_c=3;t_l=2;t_s=1;o_d=0;o_c=4;r_c=1;tl=64/dataset.pkl" --save_dir="../data/exp_icl//train_wm/name=csmdp;i_d=2;i_s=4;t_a=3;t_c=3;t_l=2;t_s=1;o_d=0;o_c=4;r_c=1;tl=64" --save_agent=True --n_iters=2000 --obj="wm" &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --name="pretrain_name=csmdp;i_d=4;i_s=4;t_a=2;t_c=0;t_l=0;t_s=3;o_d=1;o_c=1;r_c=0;tl=64" --dataset_path="../data/exp_icl//datasets/name=csmdp;i_d=4;i_s=4;t_a=2;t_c=0;t_l=0;t_s=3;o_d=1;o_c=1;r_c=0;tl=64/dataset.pkl" --save_dir="../data/exp_icl//train_wm/name=csmdp;i_d=4;i_s=4;t_a=2;t_c=0;t_l=0;t_s=3;o_d=1;o_c=1;r_c=0;tl=64" --save_agent=True --n_iters=2000 --obj="wm" &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --name="pretrain_name=csmdp;i_d=4;i_s=3;t_a=1;t_c=3;t_l=3;t_s=2;o_d=0;o_c=1;r_c=0;tl=64" --dataset_path="../data/exp_icl//datasets/name=csmdp;i_d=4;i_s=3;t_a=1;t_c=3;t_l=3;t_s=2;o_d=0;o_c=1;r_c=0;tl=64/dataset.pkl" --save_dir="../data/exp_icl//train_wm/name=csmdp;i_d=4;i_s=3;t_a=1;t_c=3;t_l=3;t_s=2;o_d=0;o_c=1;r_c=0;tl=64" --save_agent=True --n_iters=2000 --obj="wm" &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --name="pretrain_name=csmdp;i_d=1;i_s=1;t_a=4;t_c=2;t_l=2;t_s=3;o_d=0;o_c=3;r_c=2;tl=64" --dataset_path="../data/exp_icl//datasets/name=csmdp;i_d=1;i_s=1;t_a=4;t_c=2;t_l=2;t_s=3;o_d=0;o_c=3;r_c=2;tl=64/dataset.pkl" --save_dir="../data/exp_icl//train_wm/name=csmdp;i_d=1;i_s=1;t_a=4;t_c=2;t_l=2;t_s=3;o_d=0;o_c=3;r_c=2;tl=64" --save_agent=True --n_iters=2000 --obj="wm" &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --name="pretrain_name=csmdp;i_d=3;i_s=2;t_a=2;t_c=1;t_l=3;t_s=1;o_d=3;o_c=2;r_c=2;tl=64" --dataset_path="../data/exp_icl//datasets/name=csmdp;i_d=3;i_s=2;t_a=2;t_c=1;t_l=3;t_s=1;o_d=3;o_c=2;r_c=2;tl=64/dataset.pkl" --save_dir="../data/exp_icl//train_wm/name=csmdp;i_d=3;i_s=2;t_a=2;t_c=1;t_l=3;t_s=1;o_d=3;o_c=2;r_c=2;tl=64" --save_agent=True --n_iters=2000 --obj="wm" &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --name="pretrain_name=csmdp;i_d=4;i_s=1;t_a=0;t_c=3;t_l=0;t_s=4;o_d=2;o_c=1;r_c=0;tl=64" --dataset_path="../data/exp_icl//datasets/name=csmdp;i_d=4;i_s=1;t_a=0;t_c=3;t_l=0;t_s=4;o_d=2;o_c=1;r_c=0;tl=64/dataset.pkl" --save_dir="../data/exp_icl//train_wm/name=csmdp;i_d=4;i_s=1;t_a=0;t_c=3;t_l=0;t_s=4;o_d=2;o_c=1;r_c=0;tl=64" --save_agent=True --n_iters=2000 --obj="wm" &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --name="pretrain_name=csmdp;i_d=3;i_s=2;t_a=2;t_c=2;t_l=3;t_s=2;o_d=4;o_c=1;r_c=3;tl=64" --dataset_path="../data/exp_icl//datasets/name=csmdp;i_d=3;i_s=2;t_a=2;t_c=2;t_l=3;t_s=2;o_d=4;o_c=1;r_c=3;tl=64/dataset.pkl" --save_dir="../data/exp_icl//train_wm/name=csmdp;i_d=3;i_s=2;t_a=2;t_c=2;t_l=3;t_s=2;o_d=4;o_c=1;r_c=3;tl=64" --save_agent=True --n_iters=2000 --obj="wm" &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --name="pretrain_name=csmdp;i_d=0;i_s=4;t_a=3;t_c=1;t_l=3;t_s=4;o_d=4;o_c=0;r_c=2;tl=64" --dataset_path="../data/exp_icl//datasets/name=csmdp;i_d=0;i_s=4;t_a=3;t_c=1;t_l=3;t_s=4;o_d=4;o_c=0;r_c=2;tl=64/dataset.pkl" --save_dir="../data/exp_icl//train_wm/name=csmdp;i_d=0;i_s=4;t_a=3;t_c=1;t_l=3;t_s=4;o_d=4;o_c=0;r_c=2;tl=64" --save_agent=True --n_iters=2000 --obj="wm" &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --name="pretrain_name=csmdp;i_d=1;i_s=4;t_a=4;t_c=1;t_l=4;t_s=0;o_d=0;o_c=2;r_c=3;tl=64" --dataset_path="../data/exp_icl//datasets/name=csmdp;i_d=1;i_s=4;t_a=4;t_c=1;t_l=4;t_s=0;o_d=0;o_c=2;r_c=3;tl=64/dataset.pkl" --save_dir="../data/exp_icl//train_wm/name=csmdp;i_d=1;i_s=4;t_a=4;t_c=1;t_l=4;t_s=0;o_d=0;o_c=2;r_c=3;tl=64" --save_agent=True --n_iters=2000 --obj="wm" &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --name="pretrain_name=csmdp;i_d=1;i_s=0;t_a=0;t_c=0;t_l=0;t_s=2;o_d=4;o_c=0;r_c=2;tl=64" --dataset_path="../data/exp_icl//datasets/name=csmdp;i_d=1;i_s=0;t_a=0;t_c=0;t_l=0;t_s=2;o_d=4;o_c=0;r_c=2;tl=64/dataset.pkl" --save_dir="../data/exp_icl//train_wm/name=csmdp;i_d=1;i_s=0;t_a=0;t_c=0;t_l=0;t_s=2;o_d=4;o_c=0;r_c=2;tl=64" --save_agent=True --n_iters=2000 --obj="wm" &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --name="pretrain_name=csmdp;i_d=0;i_s=1;t_a=4;t_c=2;t_l=1;t_s=4;o_d=1;o_c=0;r_c=2;tl=64" --dataset_path="../data/exp_icl//datasets/name=csmdp;i_d=0;i_s=1;t_a=4;t_c=2;t_l=1;t_s=4;o_d=1;o_c=0;r_c=2;tl=64/dataset.pkl" --save_dir="../data/exp_icl//train_wm/name=csmdp;i_d=0;i_s=1;t_a=4;t_c=2;t_l=1;t_s=4;o_d=1;o_c=0;r_c=2;tl=64" --save_agent=True --n_iters=2000 --obj="wm" &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --name="pretrain_name=csmdp;i_d=3;i_s=3;t_a=3;t_c=4;t_l=1;t_s=2;o_d=2;o_c=4;r_c=2;tl=64" --dataset_path="../data/exp_icl//datasets/name=csmdp;i_d=3;i_s=3;t_a=3;t_c=4;t_l=1;t_s=2;o_d=2;o_c=4;r_c=2;tl=64/dataset.pkl" --save_dir="../data/exp_icl//train_wm/name=csmdp;i_d=3;i_s=3;t_a=3;t_c=4;t_l=1;t_s=2;o_d=2;o_c=4;r_c=2;tl=64" --save_agent=True --n_iters=2000 --obj="wm" &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --name="pretrain_name=csmdp;i_d=2;i_s=1;t_a=1;t_c=1;t_l=4;t_s=4;o_d=4;o_c=2;r_c=3;tl=64" --dataset_path="../data/exp_icl//datasets/name=csmdp;i_d=2;i_s=1;t_a=1;t_c=1;t_l=4;t_s=4;o_d=4;o_c=2;r_c=3;tl=64/dataset.pkl" --save_dir="../data/exp_icl//train_wm/name=csmdp;i_d=2;i_s=1;t_a=1;t_c=1;t_l=4;t_s=4;o_d=4;o_c=2;r_c=3;tl=64" --save_agent=True --n_iters=2000 --obj="wm" &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --name="pretrain_name=csmdp;i_d=2;i_s=1;t_a=3;t_c=0;t_l=3;t_s=1;o_d=1;o_c=1;r_c=1;tl=64" --dataset_path="../data/exp_icl//datasets/name=csmdp;i_d=2;i_s=1;t_a=3;t_c=0;t_l=3;t_s=1;o_d=1;o_c=1;r_c=1;tl=64/dataset.pkl" --save_dir="../data/exp_icl//train_wm/name=csmdp;i_d=2;i_s=1;t_a=3;t_c=0;t_l=3;t_s=1;o_d=1;o_c=1;r_c=1;tl=64" --save_agent=True --n_iters=2000 --obj="wm" &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --name="pretrain_name=csmdp;i_d=3;i_s=2;t_a=4;t_c=4;t_l=1;t_s=1;o_d=1;o_c=1;r_c=0;tl=64" --dataset_path="../data/exp_icl//datasets/name=csmdp;i_d=3;i_s=2;t_a=4;t_c=4;t_l=1;t_s=1;o_d=1;o_c=1;r_c=0;tl=64/dataset.pkl" --save_dir="../data/exp_icl//train_wm/name=csmdp;i_d=3;i_s=2;t_a=4;t_c=4;t_l=1;t_s=1;o_d=1;o_c=1;r_c=0;tl=64" --save_agent=True --n_iters=2000 --obj="wm" &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --name="pretrain_name=csmdp;i_d=1;i_s=2;t_a=0;t_c=2;t_l=1;t_s=4;o_d=0;o_c=4;r_c=2;tl=64" --dataset_path="../data/exp_icl//datasets/name=csmdp;i_d=1;i_s=2;t_a=0;t_c=2;t_l=1;t_s=4;o_d=0;o_c=4;r_c=2;tl=64/dataset.pkl" --save_dir="../data/exp_icl//train_wm/name=csmdp;i_d=1;i_s=2;t_a=0;t_c=2;t_l=1;t_s=4;o_d=0;o_c=4;r_c=2;tl=64" --save_agent=True --n_iters=2000 --obj="wm" &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --name="pretrain_name=csmdp;i_d=2;i_s=0;t_a=2;t_c=0;t_l=2;t_s=4;o_d=3;o_c=1;r_c=1;tl=64" --dataset_path="../data/exp_icl//datasets/name=csmdp;i_d=2;i_s=0;t_a=2;t_c=0;t_l=2;t_s=4;o_d=3;o_c=1;r_c=1;tl=64/dataset.pkl" --save_dir="../data/exp_icl//train_wm/name=csmdp;i_d=2;i_s=0;t_a=2;t_c=0;t_l=2;t_s=4;o_d=3;o_c=1;r_c=1;tl=64" --save_agent=True --n_iters=2000 --obj="wm" &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --name="pretrain_name=csmdp;i_d=3;i_s=3;t_a=0;t_c=2;t_l=4;t_s=1;o_d=3;o_c=3;r_c=3;tl=64" --dataset_path="../data/exp_icl//datasets/name=csmdp;i_d=3;i_s=3;t_a=0;t_c=2;t_l=4;t_s=1;o_d=3;o_c=3;r_c=3;tl=64/dataset.pkl" --save_dir="../data/exp_icl//train_wm/name=csmdp;i_d=3;i_s=3;t_a=0;t_c=2;t_l=4;t_s=1;o_d=3;o_c=3;r_c=3;tl=64" --save_agent=True --n_iters=2000 --obj="wm" &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --name="pretrain_name=csmdp;i_d=4;i_s=1;t_a=2;t_c=3;t_l=4;t_s=1;o_d=0;o_c=1;r_c=4;tl=64" --dataset_path="../data/exp_icl//datasets/name=csmdp;i_d=4;i_s=1;t_a=2;t_c=3;t_l=4;t_s=1;o_d=0;o_c=1;r_c=4;tl=64/dataset.pkl" --save_dir="../data/exp_icl//train_wm/name=csmdp;i_d=4;i_s=1;t_a=2;t_c=3;t_l=4;t_s=1;o_d=0;o_c=1;r_c=4;tl=64" --save_agent=True --n_iters=2000 --obj="wm" &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --name="pretrain_name=csmdp;i_d=2;i_s=2;t_a=0;t_c=4;t_l=2;t_s=2;o_d=4;o_c=1;r_c=3;tl=64" --dataset_path="../data/exp_icl//datasets/name=csmdp;i_d=2;i_s=2;t_a=0;t_c=4;t_l=2;t_s=2;o_d=4;o_c=1;r_c=3;tl=64/dataset.pkl" --save_dir="../data/exp_icl//train_wm/name=csmdp;i_d=2;i_s=2;t_a=0;t_c=4;t_l=2;t_s=2;o_d=4;o_c=1;r_c=3;tl=64" --save_agent=True --n_iters=2000 --obj="wm" &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --name="pretrain_name=csmdp;i_d=1;i_s=4;t_a=1;t_c=2;t_l=4;t_s=2;o_d=2;o_c=2;r_c=1;tl=64" --dataset_path="../data/exp_icl//datasets/name=csmdp;i_d=1;i_s=4;t_a=1;t_c=2;t_l=4;t_s=2;o_d=2;o_c=2;r_c=1;tl=64/dataset.pkl" --save_dir="../data/exp_icl//train_wm/name=csmdp;i_d=1;i_s=4;t_a=1;t_c=2;t_l=4;t_s=2;o_d=2;o_c=2;r_c=1;tl=64" --save_agent=True --n_iters=2000 --obj="wm" &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --name="pretrain_name=csmdp;i_d=3;i_s=0;t_a=3;t_c=0;t_l=4;t_s=3;o_d=0;o_c=2;r_c=4;tl=64" --dataset_path="../data/exp_icl//datasets/name=csmdp;i_d=3;i_s=0;t_a=3;t_c=0;t_l=4;t_s=3;o_d=0;o_c=2;r_c=4;tl=64/dataset.pkl" --save_dir="../data/exp_icl//train_wm/name=csmdp;i_d=3;i_s=0;t_a=3;t_c=0;t_l=4;t_s=3;o_d=0;o_c=2;r_c=4;tl=64" --save_agent=True --n_iters=2000 --obj="wm" &
wait
