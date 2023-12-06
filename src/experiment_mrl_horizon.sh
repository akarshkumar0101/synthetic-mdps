#!/bin/bash
source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps/bin/activate
cd /data/vision/phillipi/akumar01/synthetic-mdps/src




CUDA_VISIBLE_DEVICES=0 python run.py --n_seeds=8 --env_id="name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=1x128" --agent="linear_transformer" --run="train" --save_dir="../data/exp_mrl_horizon//train/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=1x128" --n_iters=10000 --n_envs=128 --n_envs_batch=32 --lr=0.0001 &
CUDA_VISIBLE_DEVICES=1 python run.py --n_seeds=8 --env_id="name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=8x16"  --agent="linear_transformer" --run="train" --save_dir="../data/exp_mrl_horizon//train/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=8x16"  --n_iters=10000 --n_envs=128 --n_envs_batch=32 --lr=0.0001 &
CUDA_VISIBLE_DEVICES=2 python run.py --n_seeds=8 --env_id="name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=16x8"  --agent="linear_transformer" --run="train" --save_dir="../data/exp_mrl_horizon//train/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=16x8"  --n_iters=10000 --n_envs=128 --n_envs_batch=32 --lr=0.0001 &
CUDA_VISIBLE_DEVICES=3 python run.py --n_seeds=8 --env_id="name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=128x1" --agent="linear_transformer" --run="train" --save_dir="../data/exp_mrl_horizon//train/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=128x1" --n_iters=10000 --n_envs=128 --n_envs_batch=32 --lr=0.0001 &
wait

CUDA_VISIBLE_DEVICES=0 python run.py --n_seeds=8 --env_id="name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=1x128" --agent="linear_transformer" --run="eval" --load_dir="../data/exp_mrl_horizon//train/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=1x128" --save_dir="../data/exp_mrl_horizon//eval/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=1x128/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=1x128" --n_iters=100 --n_envs=128 --n_envs_batch=32 --lr=0.0001 &
CUDA_VISIBLE_DEVICES=1 python run.py --n_seeds=8 --env_id="name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=8x16"  --agent="linear_transformer" --run="eval" --load_dir="../data/exp_mrl_horizon//train/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=1x128" --save_dir="../data/exp_mrl_horizon//eval/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=8x16/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=1x128"  --n_iters=100 --n_envs=128 --n_envs_batch=32 --lr=0.0001 &
CUDA_VISIBLE_DEVICES=2 python run.py --n_seeds=8 --env_id="name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=16x8"  --agent="linear_transformer" --run="eval" --load_dir="../data/exp_mrl_horizon//train/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=1x128" --save_dir="../data/exp_mrl_horizon//eval/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=16x8/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=1x128"  --n_iters=100 --n_envs=128 --n_envs_batch=32 --lr=0.0001 &
CUDA_VISIBLE_DEVICES=3 python run.py --n_seeds=8 --env_id="name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=128x1" --agent="linear_transformer" --run="eval" --load_dir="../data/exp_mrl_horizon//train/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=1x128" --save_dir="../data/exp_mrl_horizon//eval/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=128x1/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=1x128" --n_iters=100 --n_envs=128 --n_envs_batch=32 --lr=0.0001 &
CUDA_VISIBLE_DEVICES=4 python run.py --n_seeds=8 --env_id="name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=1x128" --agent="linear_transformer" --run="eval" --load_dir="../data/exp_mrl_horizon//train/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=8x16"  --save_dir="../data/exp_mrl_horizon//eval/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=1x128/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=8x16"  --n_iters=100 --n_envs=128 --n_envs_batch=32 --lr=0.0001 &
CUDA_VISIBLE_DEVICES=5 python run.py --n_seeds=8 --env_id="name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=8x16"  --agent="linear_transformer" --run="eval" --load_dir="../data/exp_mrl_horizon//train/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=8x16"  --save_dir="../data/exp_mrl_horizon//eval/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=8x16/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=8x16"   --n_iters=100 --n_envs=128 --n_envs_batch=32 --lr=0.0001 &
CUDA_VISIBLE_DEVICES=6 python run.py --n_seeds=8 --env_id="name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=16x8"  --agent="linear_transformer" --run="eval" --load_dir="../data/exp_mrl_horizon//train/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=8x16"  --save_dir="../data/exp_mrl_horizon//eval/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=16x8/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=8x16"   --n_iters=100 --n_envs=128 --n_envs_batch=32 --lr=0.0001 &
CUDA_VISIBLE_DEVICES=7 python run.py --n_seeds=8 --env_id="name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=128x1" --agent="linear_transformer" --run="eval" --load_dir="../data/exp_mrl_horizon//train/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=8x16"  --save_dir="../data/exp_mrl_horizon//eval/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=128x1/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=8x16"  --n_iters=100 --n_envs=128 --n_envs_batch=32 --lr=0.0001 &
wait
CUDA_VISIBLE_DEVICES=0 python run.py --n_seeds=8 --env_id="name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=1x128" --agent="linear_transformer" --run="eval" --load_dir="../data/exp_mrl_horizon//train/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=16x8"  --save_dir="../data/exp_mrl_horizon//eval/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=1x128/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=16x8"  --n_iters=100 --n_envs=128 --n_envs_batch=32 --lr=0.0001 &
CUDA_VISIBLE_DEVICES=1 python run.py --n_seeds=8 --env_id="name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=8x16"  --agent="linear_transformer" --run="eval" --load_dir="../data/exp_mrl_horizon//train/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=16x8"  --save_dir="../data/exp_mrl_horizon//eval/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=8x16/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=16x8"   --n_iters=100 --n_envs=128 --n_envs_batch=32 --lr=0.0001 &
CUDA_VISIBLE_DEVICES=2 python run.py --n_seeds=8 --env_id="name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=16x8"  --agent="linear_transformer" --run="eval" --load_dir="../data/exp_mrl_horizon//train/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=16x8"  --save_dir="../data/exp_mrl_horizon//eval/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=16x8/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=16x8"   --n_iters=100 --n_envs=128 --n_envs_batch=32 --lr=0.0001 &
CUDA_VISIBLE_DEVICES=3 python run.py --n_seeds=8 --env_id="name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=128x1" --agent="linear_transformer" --run="eval" --load_dir="../data/exp_mrl_horizon//train/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=16x8"  --save_dir="../data/exp_mrl_horizon//eval/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=128x1/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=16x8"  --n_iters=100 --n_envs=128 --n_envs_batch=32 --lr=0.0001 &
CUDA_VISIBLE_DEVICES=4 python run.py --n_seeds=8 --env_id="name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=1x128" --agent="linear_transformer" --run="eval" --load_dir="../data/exp_mrl_horizon//train/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=128x1" --save_dir="../data/exp_mrl_horizon//eval/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=1x128/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=128x1" --n_iters=100 --n_envs=128 --n_envs_batch=32 --lr=0.0001 &
CUDA_VISIBLE_DEVICES=5 python run.py --n_seeds=8 --env_id="name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=8x16"  --agent="linear_transformer" --run="eval" --load_dir="../data/exp_mrl_horizon//train/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=128x1" --save_dir="../data/exp_mrl_horizon//eval/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=8x16/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=128x1"  --n_iters=100 --n_envs=128 --n_envs_batch=32 --lr=0.0001 &
CUDA_VISIBLE_DEVICES=6 python run.py --n_seeds=8 --env_id="name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=16x8"  --agent="linear_transformer" --run="eval" --load_dir="../data/exp_mrl_horizon//train/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=128x1" --save_dir="../data/exp_mrl_horizon//eval/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=16x8/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=128x1"  --n_iters=100 --n_envs=128 --n_envs_batch=32 --lr=0.0001 &
CUDA_VISIBLE_DEVICES=7 python run.py --n_seeds=8 --env_id="name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=128x1" --agent="linear_transformer" --run="eval" --load_dir="../data/exp_mrl_horizon//train/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=128x1" --save_dir="../data/exp_mrl_horizon//eval/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=128x1/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=128x1" --n_iters=100 --n_envs=128 --n_envs_batch=32 --lr=0.0001 &
wait
CUDA_VISIBLE_DEVICES=0 python run.py --n_seeds=8 --env_id="name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=1x128" --agent="linear_transformer" --run="eval" --load_dir=None                                                                                       --save_dir="../data/exp_mrl_horizon//eval/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=1x128/None"                                                      --n_iters=100 --n_envs=128 --n_envs_batch=32 --lr=0.0001 &
CUDA_VISIBLE_DEVICES=1 python run.py --n_seeds=8 --env_id="name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=8x16"  --agent="linear_transformer" --run="eval" --load_dir=None                                                                                       --save_dir="../data/exp_mrl_horizon//eval/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=8x16/None"                                                       --n_iters=100 --n_envs=128 --n_envs_batch=32 --lr=0.0001 &
CUDA_VISIBLE_DEVICES=2 python run.py --n_seeds=8 --env_id="name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=16x8"  --agent="linear_transformer" --run="eval" --load_dir=None                                                                                       --save_dir="../data/exp_mrl_horizon//eval/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=16x8/None"                                                       --n_iters=100 --n_envs=128 --n_envs_batch=32 --lr=0.0001 &
CUDA_VISIBLE_DEVICES=3 python run.py --n_seeds=8 --env_id="name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=128x1" --agent="linear_transformer" --run="eval" --load_dir=None                                                                                       --save_dir="../data/exp_mrl_horizon//eval/name=dsmdp;n_states=64;n_acts=4;d_obs=16;rpo=16;mrl=128x1/None"                                                      --n_iters=100 --n_envs=128 --n_envs_batch=32 --lr=0.0001 &
wait
