python icl_train.py --seed=0 --load_ckpt="/vision-nfs/isola/env/akumar01/synthetic-mdps-data//icl_train/Reacher_gato_aug/ckpt_latest.pkl"                --save_dir="/vision-nfs/isola/env/akumar01/synthetic-mdps-data//icl_test/Reacher/Reacher_gato_aug"                               --save_ckpt=False --dataset_paths="/vision-nfs/isola/env/akumar01/synthetic-mdps-data//datasets/mujoco/Reacher/dataset.pkl"                --exclude_dataset_paths=None --n_augs=0 --aug_dist="uniform" --nv=1 --nh=256 --n_iters=0 --bs=128 --lr=0.0003 --lr_schedule="constant" --weight_decay=0.0 --clip_grad_norm=1.0 --d_obs_uni=32 --d_act_uni=8 --n_layers=4 --n_heads=8 --d_embd=256 --ctx_len=128 --mask_type="causal" --env_id="Reacher"                --n_envs=64 --n_iters_rollout=1000 --video=False
python icl_train.py --seed=0 --load_ckpt="/vision-nfs/isola/env/akumar01/synthetic-mdps-data//icl_train/InvertedPendulum_gato_aug/ckpt_latest.pkl"       --save_dir="/vision-nfs/isola/env/akumar01/synthetic-mdps-data//icl_test/InvertedPendulum/InvertedPendulum_gato_aug"             --save_ckpt=False --dataset_paths="/vision-nfs/isola/env/akumar01/synthetic-mdps-data//datasets/mujoco/InvertedPendulum/dataset.pkl"       --exclude_dataset_paths=None --n_augs=0 --aug_dist="uniform" --nv=1 --nh=256 --n_iters=0 --bs=128 --lr=0.0003 --lr_schedule="constant" --weight_decay=0.0 --clip_grad_norm=1.0 --d_obs_uni=32 --d_act_uni=8 --n_layers=4 --n_heads=8 --d_embd=256 --ctx_len=128 --mask_type="causal" --env_id="InvertedPendulum"       --n_envs=64 --n_iters_rollout=1000 --video=False
python icl_train.py --seed=0 --load_ckpt="/vision-nfs/isola/env/akumar01/synthetic-mdps-data//icl_train/InvertedDoublePendulum_gato_aug/ckpt_latest.pkl" --save_dir="/vision-nfs/isola/env/akumar01/synthetic-mdps-data//icl_test/InvertedDoublePendulum/InvertedDoublePendulum_gato_aug" --save_ckpt=False --dataset_paths="/vision-nfs/isola/env/akumar01/synthetic-mdps-data//datasets/mujoco/InvertedDoublePendulum/dataset.pkl" --exclude_dataset_paths=None --n_augs=0 --aug_dist="uniform" --nv=1 --nh=256 --n_iters=0 --bs=128 --lr=0.0003 --lr_schedule="constant" --weight_decay=0.0 --clip_grad_norm=1.0 --d_obs_uni=32 --d_act_uni=8 --n_layers=4 --n_heads=8 --d_embd=256 --ctx_len=128 --mask_type="causal" --env_id="InvertedDoublePendulum" --n_envs=64 --n_iters_rollout=1000 --video=False
python icl_train.py --seed=0 --load_ckpt="/vision-nfs/isola/env/akumar01/synthetic-mdps-data//icl_train/HalfCheetah_gato_aug/ckpt_latest.pkl"            --save_dir="/vision-nfs/isola/env/akumar01/synthetic-mdps-data//icl_test/HalfCheetah/HalfCheetah_gato_aug"                       --save_ckpt=False --dataset_paths="/vision-nfs/isola/env/akumar01/synthetic-mdps-data//datasets/mujoco/HalfCheetah/dataset.pkl"            --exclude_dataset_paths=None --n_augs=0 --aug_dist="uniform" --nv=1 --nh=256 --n_iters=0 --bs=128 --lr=0.0003 --lr_schedule="constant" --weight_decay=0.0 --clip_grad_norm=1.0 --d_obs_uni=32 --d_act_uni=8 --n_layers=4 --n_heads=8 --d_embd=256 --ctx_len=128 --mask_type="causal" --env_id="HalfCheetah"            --n_envs=64 --n_iters_rollout=1000 --video=False
python icl_train.py --seed=0 --load_ckpt="/vision-nfs/isola/env/akumar01/synthetic-mdps-data//icl_train/Hopper_gato_aug/ckpt_latest.pkl"                 --save_dir="/vision-nfs/isola/env/akumar01/synthetic-mdps-data//icl_test/Hopper/Hopper_gato_aug"                                 --save_ckpt=False --dataset_paths="/vision-nfs/isola/env/akumar01/synthetic-mdps-data//datasets/mujoco/Hopper/dataset.pkl"                 --exclude_dataset_paths=None --n_augs=0 --aug_dist="uniform" --nv=1 --nh=256 --n_iters=0 --bs=128 --lr=0.0003 --lr_schedule="constant" --weight_decay=0.0 --clip_grad_norm=1.0 --d_obs_uni=32 --d_act_uni=8 --n_layers=4 --n_heads=8 --d_embd=256 --ctx_len=128 --mask_type="causal" --env_id="Hopper"                 --n_envs=64 --n_iters_rollout=1000 --video=False
python icl_train.py --seed=0 --load_ckpt="/vision-nfs/isola/env/akumar01/synthetic-mdps-data//icl_train/Walker2d_gato_aug/ckpt_latest.pkl"               --save_dir="/vision-nfs/isola/env/akumar01/synthetic-mdps-data//icl_test/Walker2d/Walker2d_gato_aug"                             --save_ckpt=False --dataset_paths="/vision-nfs/isola/env/akumar01/synthetic-mdps-data//datasets/mujoco/Walker2d/dataset.pkl"               --exclude_dataset_paths=None --n_augs=0 --aug_dist="uniform" --nv=1 --nh=256 --n_iters=0 --bs=128 --lr=0.0003 --lr_schedule="constant" --weight_decay=0.0 --clip_grad_norm=1.0 --d_obs_uni=32 --d_act_uni=8 --n_layers=4 --n_heads=8 --d_embd=256 --ctx_len=128 --mask_type="causal" --env_id="Walker2d"               --n_envs=64 --n_iters_rollout=1000 --video=False
python icl_train.py --seed=0 --load_ckpt="/vision-nfs/isola/env/akumar01/synthetic-mdps-data//icl_train/Ant_gato_aug/ckpt_latest.pkl"                    --save_dir="/vision-nfs/isola/env/akumar01/synthetic-mdps-data//icl_test/Ant/Ant_gato_aug"                                       --save_ckpt=False --dataset_paths="/vision-nfs/isola/env/akumar01/synthetic-mdps-data//datasets/mujoco/Ant/dataset.pkl"                    --exclude_dataset_paths=None --n_augs=0 --aug_dist="uniform" --nv=1 --nh=256 --n_iters=0 --bs=128 --lr=0.0003 --lr_schedule="constant" --weight_decay=0.0 --clip_grad_norm=1.0 --d_obs_uni=32 --d_act_uni=8 --n_layers=4 --n_heads=8 --d_embd=256 --ctx_len=128 --mask_type="causal" --env_id="Ant"                    --n_envs=64 --n_iters_rollout=1000 --video=False
