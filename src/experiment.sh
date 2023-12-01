CUDA_VISIBLE_DEVICES=0 python run.py --n_seeds=8 --env_id="env=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=1"   --agent="linear_transformer" --run="train" --load_dir=None --save_dir="../data/env=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=1"   --n_iters=1000 &
CUDA_VISIBLE_DEVICES=1 python run.py --n_seeds=8 --env_id="env=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=128" --agent="linear_transformer" --run="train" --load_dir=None --save_dir="../data/env=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=128" --n_iters=1000 &
wait




CUDA_VISIBLE_DEVICES=0 python run.py --n_seeds=8 --env_id="env=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=1"   --agent="linear_transformer" --run="eval" --load_dir="../data/env=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=1"   --save_dir="../data/transfer/env=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=1/env=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=1"     --n_iters=10 &
CUDA_VISIBLE_DEVICES=1 python run.py --n_seeds=8 --env_id="env=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=128" --agent="linear_transformer" --run="eval" --load_dir="../data/env=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=1"   --save_dir="../data/transfer/env=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=128/env=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=1"   --n_iters=10 &
CUDA_VISIBLE_DEVICES=2 python run.py --n_seeds=8 --env_id="env=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=1"   --agent="linear_transformer" --run="eval" --load_dir="../data/env=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=128" --save_dir="../data/transfer/env=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=1/env=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=128"   --n_iters=10 &
CUDA_VISIBLE_DEVICES=3 python run.py --n_seeds=8 --env_id="env=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=128" --agent="linear_transformer" --run="eval" --load_dir="../data/env=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=128" --save_dir="../data/transfer/env=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=128/env=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=128" --n_iters=10 &
wait
wait