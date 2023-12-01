#!/bin/bash
source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps/bin/activate
cd /data/vision/phillipi/akumar01/synthetic-mdps/src



CUDA_VISIBLE_DEVICES=0 python run.py --n_seeds=8 --env_id="name=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=1"   --agent="linear_transformer" --run="train" --save_dir="../data/name=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=1"   --n_iters=10 &
CUDA_VISIBLE_DEVICES=1 python run.py --n_seeds=8 --env_id="name=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=128" --agent="linear_transformer" --run="train" --save_dir="../data/name=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=128" --n_iters=10 &
wait




CUDA_VISIBLE_DEVICES=0 python run.py --n_seeds=8 --env_id="name=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=1"   --agent="linear_transformer" --run="eval" --load_dir="../data/name=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=1"   --save_dir="../data/transfer/name=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=1/name=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=1"     --n_iters=10 &
CUDA_VISIBLE_DEVICES=1 python run.py --n_seeds=8 --env_id="name=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=128" --agent="linear_transformer" --run="eval" --load_dir="../data/name=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=1"   --save_dir="../data/transfer/name=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=128/name=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=1"   --n_iters=10 &
CUDA_VISIBLE_DEVICES=2 python run.py --n_seeds=8 --env_id="name=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=1"   --agent="linear_transformer" --run="eval" --load_dir="../data/name=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=128" --save_dir="../data/transfer/name=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=1/name=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=128"   --n_iters=10 &
CUDA_VISIBLE_DEVICES=3 python run.py --n_seeds=8 --env_id="name=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=128" --agent="linear_transformer" --run="eval" --load_dir="../data/name=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=128" --save_dir="../data/transfer/name=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=128/name=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=128" --n_iters=10 &
wait