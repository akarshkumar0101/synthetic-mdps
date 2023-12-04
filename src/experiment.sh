#!/bin/bash
source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps/bin/activate
cd /data/vision/phillipi/akumar01/synthetic-mdps/src




CUDA_VISIBLE_DEVICES=0 python run.py --n_seeds=8 --env_id="name=cartpole;fobs=T;tl=128"    --agent="linear_transformer" --run="train" --save_dir="../data/train/name=cartpole;fobs=T;tl=128"    --n_iters=1000 &
CUDA_VISIBLE_DEVICES=1 python run.py --n_seeds=8 --env_id="name=mountaincar;fobs=T;tl=128" --agent="linear_transformer" --run="train" --save_dir="../data/train/name=mountaincar;fobs=T;tl=128" --n_iters=1000 &
CUDA_VISIBLE_DEVICES=2 python run.py --n_seeds=8 --env_id="name=acrobot;fobs=T;tl=128"     --agent="linear_transformer" --run="train" --save_dir="../data/train/name=acrobot;fobs=T;tl=128"     --n_iters=1000 &
wait

CUDA_VISIBLE_DEVICES=0 python run.py --n_seeds=8 --env_id="name=cartpole;fobs=T;tl=128"    --agent="linear_transformer" --run="eval" --load_dir="../data/train/name=cartpole;fobs=T;tl=128"    --save_dir="../data/eval/name=cartpole;fobs=T;tl=128/name=cartpole;fobs=T;tl=128"       --n_iters=10 &
CUDA_VISIBLE_DEVICES=1 python run.py --n_seeds=8 --env_id="name=mountaincar;fobs=T;tl=128" --agent="linear_transformer" --run="eval" --load_dir="../data/train/name=mountaincar;fobs=T;tl=128" --save_dir="../data/eval/name=mountaincar;fobs=T;tl=128/name=mountaincar;fobs=T;tl=128" --n_iters=10 &
CUDA_VISIBLE_DEVICES=2 python run.py --n_seeds=8 --env_id="name=acrobot;fobs=T;tl=128"     --agent="linear_transformer" --run="eval" --load_dir="../data/train/name=acrobot;fobs=T;tl=128"     --save_dir="../data/eval/name=acrobot;fobs=T;tl=128/name=acrobot;fobs=T;tl=128"         --n_iters=10 &
wait
