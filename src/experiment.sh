#!/bin/bash
source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps/bin/activate
cd /data/vision/phillipi/akumar01/synthetic-mdps/src





CUDA_VISIBLE_DEVICES=0 python icl_bc.py --name="pretrain_CartPole-v1_time_perm_True"  --dataset_path="../data/temp/expert_data_CartPole-v1.pkl" --save_dir="../data/exp_iclbc//pretrain/CartPole-v1_True"  --time_perm=True  &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --name="pretrain_Acrobot-v1_time_perm_True"   --dataset_path="../data/temp/expert_data_Acrobot-v1.pkl"  --save_dir="../data/exp_iclbc//pretrain/Acrobot-v1_True"   --time_perm=True  &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --name="pretrain_synthetic_time_perm_True"    --dataset_path="../data/temp/expert_data_synthetic.pkl"   --save_dir="../data/exp_iclbc//pretrain/synthetic_True"    --time_perm=True  &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --name="pretrain_CartPole-v1_time_perm_False" --dataset_path="../data/temp/expert_data_CartPole-v1.pkl" --save_dir="../data/exp_iclbc//pretrain/CartPole-v1_False" --time_perm=False &
CUDA_VISIBLE_DEVICES=4 python icl_bc.py --name="pretrain_Acrobot-v1_time_perm_False"  --dataset_path="../data/temp/expert_data_Acrobot-v1.pkl"  --save_dir="../data/exp_iclbc//pretrain/Acrobot-v1_False"  --time_perm=False &
CUDA_VISIBLE_DEVICES=5 python icl_bc.py --name="pretrain_synthetic_time_perm_False"   --dataset_path="../data/temp/expert_data_synthetic.pkl"   --save_dir="../data/exp_iclbc//pretrain/synthetic_False"   --time_perm=False &
wait

CUDA_VISIBLE_DEVICES=0 python icl_bc.py --name="train_CartPole-v1_test_CartPole-v1_time_perm_True"  --dataset_path="../data/temp/expert_data_CartPole-v1.pkl" --load_dir="../data/exp_iclbc//pretrain/CartPole-v1_True"  --save_dir="../data/exp_iclbc//eval/CartPole-v1/CartPole-v1_True"  --n_iters=1000 --time_perm=True  &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --name="train_CartPole-v1_test_Acrobot-v1_time_perm_True"   --dataset_path="../data/temp/expert_data_Acrobot-v1.pkl"  --load_dir="../data/exp_iclbc//pretrain/CartPole-v1_True"  --save_dir="../data/exp_iclbc//eval/CartPole-v1/Acrobot-v1_True"   --n_iters=1000 --time_perm=True  &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --name="train_CartPole-v1_test_synthetic_time_perm_True"    --dataset_path="../data/temp/expert_data_synthetic.pkl"   --load_dir="../data/exp_iclbc//pretrain/CartPole-v1_True"  --save_dir="../data/exp_iclbc//eval/CartPole-v1/synthetic_True"    --n_iters=1000 --time_perm=True  &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --name="train_Acrobot-v1_test_CartPole-v1_time_perm_True"   --dataset_path="../data/temp/expert_data_CartPole-v1.pkl" --load_dir="../data/exp_iclbc//pretrain/Acrobot-v1_True"   --save_dir="../data/exp_iclbc//eval/Acrobot-v1/CartPole-v1_True"   --n_iters=1000 --time_perm=True  &
CUDA_VISIBLE_DEVICES=4 python icl_bc.py --name="train_Acrobot-v1_test_Acrobot-v1_time_perm_True"    --dataset_path="../data/temp/expert_data_Acrobot-v1.pkl"  --load_dir="../data/exp_iclbc//pretrain/Acrobot-v1_True"   --save_dir="../data/exp_iclbc//eval/Acrobot-v1/Acrobot-v1_True"    --n_iters=1000 --time_perm=True  &
CUDA_VISIBLE_DEVICES=5 python icl_bc.py --name="train_Acrobot-v1_test_synthetic_time_perm_True"     --dataset_path="../data/temp/expert_data_synthetic.pkl"   --load_dir="../data/exp_iclbc//pretrain/Acrobot-v1_True"   --save_dir="../data/exp_iclbc//eval/Acrobot-v1/synthetic_True"     --n_iters=1000 --time_perm=True  &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --name="train_synthetic_test_CartPole-v1_time_perm_True"    --dataset_path="../data/temp/expert_data_CartPole-v1.pkl" --load_dir="../data/exp_iclbc//pretrain/synthetic_True"    --save_dir="../data/exp_iclbc//eval/synthetic/CartPole-v1_True"    --n_iters=1000 --time_perm=True  &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --name="train_synthetic_test_Acrobot-v1_time_perm_True"     --dataset_path="../data/temp/expert_data_Acrobot-v1.pkl"  --load_dir="../data/exp_iclbc//pretrain/synthetic_True"    --save_dir="../data/exp_iclbc//eval/synthetic/Acrobot-v1_True"     --n_iters=1000 --time_perm=True  &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --name="train_synthetic_test_synthetic_time_perm_True"      --dataset_path="../data/temp/expert_data_synthetic.pkl"   --load_dir="../data/exp_iclbc//pretrain/synthetic_True"    --save_dir="../data/exp_iclbc//eval/synthetic/synthetic_True"      --n_iters=1000 --time_perm=True  &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --name="train_CartPole-v1_test_CartPole-v1_time_perm_False" --dataset_path="../data/temp/expert_data_CartPole-v1.pkl" --load_dir="../data/exp_iclbc//pretrain/CartPole-v1_False" --save_dir="../data/exp_iclbc//eval/CartPole-v1/CartPole-v1_False" --n_iters=1000 --time_perm=False &
CUDA_VISIBLE_DEVICES=4 python icl_bc.py --name="train_CartPole-v1_test_Acrobot-v1_time_perm_False"  --dataset_path="../data/temp/expert_data_Acrobot-v1.pkl"  --load_dir="../data/exp_iclbc//pretrain/CartPole-v1_False" --save_dir="../data/exp_iclbc//eval/CartPole-v1/Acrobot-v1_False"  --n_iters=1000 --time_perm=False &
CUDA_VISIBLE_DEVICES=5 python icl_bc.py --name="train_CartPole-v1_test_synthetic_time_perm_False"   --dataset_path="../data/temp/expert_data_synthetic.pkl"   --load_dir="../data/exp_iclbc//pretrain/CartPole-v1_False" --save_dir="../data/exp_iclbc//eval/CartPole-v1/synthetic_False"   --n_iters=1000 --time_perm=False &
wait
CUDA_VISIBLE_DEVICES=0 python icl_bc.py --name="train_Acrobot-v1_test_CartPole-v1_time_perm_False"  --dataset_path="../data/temp/expert_data_CartPole-v1.pkl" --load_dir="../data/exp_iclbc//pretrain/Acrobot-v1_False"  --save_dir="../data/exp_iclbc//eval/Acrobot-v1/CartPole-v1_False"  --n_iters=1000 --time_perm=False &
CUDA_VISIBLE_DEVICES=1 python icl_bc.py --name="train_Acrobot-v1_test_Acrobot-v1_time_perm_False"   --dataset_path="../data/temp/expert_data_Acrobot-v1.pkl"  --load_dir="../data/exp_iclbc//pretrain/Acrobot-v1_False"  --save_dir="../data/exp_iclbc//eval/Acrobot-v1/Acrobot-v1_False"   --n_iters=1000 --time_perm=False &
CUDA_VISIBLE_DEVICES=2 python icl_bc.py --name="train_Acrobot-v1_test_synthetic_time_perm_False"    --dataset_path="../data/temp/expert_data_synthetic.pkl"   --load_dir="../data/exp_iclbc//pretrain/Acrobot-v1_False"  --save_dir="../data/exp_iclbc//eval/Acrobot-v1/synthetic_False"    --n_iters=1000 --time_perm=False &
CUDA_VISIBLE_DEVICES=3 python icl_bc.py --name="train_synthetic_test_CartPole-v1_time_perm_False"   --dataset_path="../data/temp/expert_data_CartPole-v1.pkl" --load_dir="../data/exp_iclbc//pretrain/synthetic_False"   --save_dir="../data/exp_iclbc//eval/synthetic/CartPole-v1_False"   --n_iters=1000 --time_perm=False &
CUDA_VISIBLE_DEVICES=4 python icl_bc.py --name="train_synthetic_test_Acrobot-v1_time_perm_False"    --dataset_path="../data/temp/expert_data_Acrobot-v1.pkl"  --load_dir="../data/exp_iclbc//pretrain/synthetic_False"   --save_dir="../data/exp_iclbc//eval/synthetic/Acrobot-v1_False"    --n_iters=1000 --time_perm=False &
CUDA_VISIBLE_DEVICES=5 python icl_bc.py --name="train_synthetic_test_synthetic_time_perm_False"     --dataset_path="../data/temp/expert_data_synthetic.pkl"   --load_dir="../data/exp_iclbc//pretrain/synthetic_False"   --save_dir="../data/exp_iclbc//eval/synthetic/synthetic_False"     --n_iters=1000 --time_perm=False &
wait
