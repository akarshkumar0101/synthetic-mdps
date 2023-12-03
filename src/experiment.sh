#!/bin/bash
source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps/bin/activate
cd /data/vision/phillipi/akumar01/synthetic-mdps/src


#CUDA_VISIBLE_DEVICES=0 python run.py --n_seeds=8 --env_id="name=gridenv;grid_len=8;pos_start=fixed;pos_rew=fixed;fobs=T;tl=128" --agent="linear_transformer" --run="train" --save_dir="../data/name=gridenv;grid_len=8;pos_start=fixed;pos_rew=fixed;fobs=T;tl=128" --n_iters=1000 &
#wait
#CUDA_VISIBLE_DEVICES=0 python run.py --n_seeds=8 --env_id="name=gridenv;grid_len=8;pos_start=fixed;pos_rew=fixed;fobs=T;tl=128" --agent="linear_transformer" --run="eval" --load_dir="../data/name=gridenv;grid_len=8;pos_start=fixed;pos_rew=fixed;fobs=T;tl=128" --save_dir="../data/transfer/name=gridenv;grid_len=8;pos_start=fixed;pos_rew=fixed;fobs=T;tl=128/name=gridenv;grid_len=8;pos_start=fixed;pos_rew=fixed;fobs=T;tl=128" --n_iters=10 &
#wait


CUDA_VISIBLE_DEVICES=0 python run.py --n_seeds=8 --env_id="name=gridenv;grid_len=8;pos_start=random;pos_rew=random;fobs=T;tl=128" --agent="linear_transformer" --run="train" --save_dir="../data/name=gridenv;grid_len=8;pos_start=random;pos_rew=random;fobs=T;tl=128" --n_iters=1000 &
wait
CUDA_VISIBLE_DEVICES=0 python run.py --n_seeds=8 --env_id="name=gridenv;grid_len=8;pos_start=random;pos_rew=random;fobs=T;tl=128" --agent="linear_transformer" --run="eval" --load_dir="../data/name=gridenv;grid_len=8;pos_start=random;pos_rew=random;fobs=T;tl=128" --save_dir="../data/transfer/name=gridenv;grid_len=8;pos_start=random;pos_rew=random;fobs=T;tl=128/name=gridenv;grid_len=8;pos_start=random;pos_rew=random;fobs=T;tl=128" --n_iters=10 &
wait
