python icl_bc_ed.py --seed=0 --load_ckpt=None --save_dir="/data/vision/phillipi/akumar01/synthetic-mdps-data/train_bc//mujoco/all-Reacher"                --n_ckpts=1 --obj="bc" --dataset_paths="/data/vision/phillipi/akumar01/synthetic-mdps-data/datasets/mujoco/*/dataset.pkl" --exclude_dataset_paths="/data/vision/phillipi/akumar01/synthetic-mdps-data/datasets/mujoco/Reacher/dataset.pkl"                --n_augs=0 --aug_dist="uniform" --nv=4096 --nh=131072 --n_iters_eval=10 --n_iters=50000 --bs=64 --mini_bs=64 --lr=0.0003 --lr_schedule="constant" --weight_decay=0.0 --clip_grad_norm=1.0 --d_obs_uni=128 --d_act_uni=21 --n_layers=4 --n_heads=8 --d_embd=256 --ctx_len=256 --seq_len=256 --mask_type="causal" --env_id=None --n_iters_rollout=100 --percent_data=0.25
python icl_bc_ed.py --seed=0 --load_ckpt=None --save_dir="/data/vision/phillipi/akumar01/synthetic-mdps-data/train_bc//mujoco/all-Pusher"                 --n_ckpts=1 --obj="bc" --dataset_paths="/data/vision/phillipi/akumar01/synthetic-mdps-data/datasets/mujoco/*/dataset.pkl" --exclude_dataset_paths="/data/vision/phillipi/akumar01/synthetic-mdps-data/datasets/mujoco/Pusher/dataset.pkl"                 --n_augs=0 --aug_dist="uniform" --nv=4096 --nh=131072 --n_iters_eval=10 --n_iters=50000 --bs=64 --mini_bs=64 --lr=0.0003 --lr_schedule="constant" --weight_decay=0.0 --clip_grad_norm=1.0 --d_obs_uni=128 --d_act_uni=21 --n_layers=4 --n_heads=8 --d_embd=256 --ctx_len=256 --seq_len=256 --mask_type="causal" --env_id=None --n_iters_rollout=100 --percent_data=0.25
python icl_bc_ed.py --seed=0 --load_ckpt=None --save_dir="/data/vision/phillipi/akumar01/synthetic-mdps-data/train_bc//mujoco/all-InvertedPendulum"       --n_ckpts=1 --obj="bc" --dataset_paths="/data/vision/phillipi/akumar01/synthetic-mdps-data/datasets/mujoco/*/dataset.pkl" --exclude_dataset_paths="/data/vision/phillipi/akumar01/synthetic-mdps-data/datasets/mujoco/InvertedPendulum/dataset.pkl"       --n_augs=0 --aug_dist="uniform" --nv=4096 --nh=131072 --n_iters_eval=10 --n_iters=50000 --bs=64 --mini_bs=64 --lr=0.0003 --lr_schedule="constant" --weight_decay=0.0 --clip_grad_norm=1.0 --d_obs_uni=128 --d_act_uni=21 --n_layers=4 --n_heads=8 --d_embd=256 --ctx_len=256 --seq_len=256 --mask_type="causal" --env_id=None --n_iters_rollout=100 --percent_data=0.25
python icl_bc_ed.py --seed=0 --load_ckpt=None --save_dir="/data/vision/phillipi/akumar01/synthetic-mdps-data/train_bc//mujoco/all-InvertedDoublePendulum" --n_ckpts=1 --obj="bc" --dataset_paths="/data/vision/phillipi/akumar01/synthetic-mdps-data/datasets/mujoco/*/dataset.pkl" --exclude_dataset_paths="/data/vision/phillipi/akumar01/synthetic-mdps-data/datasets/mujoco/InvertedDoublePendulum/dataset.pkl" --n_augs=0 --aug_dist="uniform" --nv=4096 --nh=131072 --n_iters_eval=10 --n_iters=50000 --bs=64 --mini_bs=64 --lr=0.0003 --lr_schedule="constant" --weight_decay=0.0 --clip_grad_norm=1.0 --d_obs_uni=128 --d_act_uni=21 --n_layers=4 --n_heads=8 --d_embd=256 --ctx_len=256 --seq_len=256 --mask_type="causal" --env_id=None --n_iters_rollout=100 --percent_data=0.25
python icl_bc_ed.py --seed=0 --load_ckpt=None --save_dir="/data/vision/phillipi/akumar01/synthetic-mdps-data/train_bc//mujoco/all-HalfCheetah"            --n_ckpts=1 --obj="bc" --dataset_paths="/data/vision/phillipi/akumar01/synthetic-mdps-data/datasets/mujoco/*/dataset.pkl" --exclude_dataset_paths="/data/vision/phillipi/akumar01/synthetic-mdps-data/datasets/mujoco/HalfCheetah/dataset.pkl"            --n_augs=0 --aug_dist="uniform" --nv=4096 --nh=131072 --n_iters_eval=10 --n_iters=50000 --bs=64 --mini_bs=64 --lr=0.0003 --lr_schedule="constant" --weight_decay=0.0 --clip_grad_norm=1.0 --d_obs_uni=128 --d_act_uni=21 --n_layers=4 --n_heads=8 --d_embd=256 --ctx_len=256 --seq_len=256 --mask_type="causal" --env_id=None --n_iters_rollout=100 --percent_data=0.25
python icl_bc_ed.py --seed=0 --load_ckpt=None --save_dir="/data/vision/phillipi/akumar01/synthetic-mdps-data/train_bc//mujoco/all-Hopper"                 --n_ckpts=1 --obj="bc" --dataset_paths="/data/vision/phillipi/akumar01/synthetic-mdps-data/datasets/mujoco/*/dataset.pkl" --exclude_dataset_paths="/data/vision/phillipi/akumar01/synthetic-mdps-data/datasets/mujoco/Hopper/dataset.pkl"                 --n_augs=0 --aug_dist="uniform" --nv=4096 --nh=131072 --n_iters_eval=10 --n_iters=50000 --bs=64 --mini_bs=64 --lr=0.0003 --lr_schedule="constant" --weight_decay=0.0 --clip_grad_norm=1.0 --d_obs_uni=128 --d_act_uni=21 --n_layers=4 --n_heads=8 --d_embd=256 --ctx_len=256 --seq_len=256 --mask_type="causal" --env_id=None --n_iters_rollout=100 --percent_data=0.25
python icl_bc_ed.py --seed=0 --load_ckpt=None --save_dir="/data/vision/phillipi/akumar01/synthetic-mdps-data/train_bc//mujoco/all-Swimmer"                --n_ckpts=1 --obj="bc" --dataset_paths="/data/vision/phillipi/akumar01/synthetic-mdps-data/datasets/mujoco/*/dataset.pkl" --exclude_dataset_paths="/data/vision/phillipi/akumar01/synthetic-mdps-data/datasets/mujoco/Swimmer/dataset.pkl"                --n_augs=0 --aug_dist="uniform" --nv=4096 --nh=131072 --n_iters_eval=10 --n_iters=50000 --bs=64 --mini_bs=64 --lr=0.0003 --lr_schedule="constant" --weight_decay=0.0 --clip_grad_norm=1.0 --d_obs_uni=128 --d_act_uni=21 --n_layers=4 --n_heads=8 --d_embd=256 --ctx_len=256 --seq_len=256 --mask_type="causal" --env_id=None --n_iters_rollout=100 --percent_data=0.25
python icl_bc_ed.py --seed=0 --load_ckpt=None --save_dir="/data/vision/phillipi/akumar01/synthetic-mdps-data/train_bc//mujoco/all-Walker2d"               --n_ckpts=1 --obj="bc" --dataset_paths="/data/vision/phillipi/akumar01/synthetic-mdps-data/datasets/mujoco/*/dataset.pkl" --exclude_dataset_paths="/data/vision/phillipi/akumar01/synthetic-mdps-data/datasets/mujoco/Walker2d/dataset.pkl"               --n_augs=0 --aug_dist="uniform" --nv=4096 --nh=131072 --n_iters_eval=10 --n_iters=50000 --bs=64 --mini_bs=64 --lr=0.0003 --lr_schedule="constant" --weight_decay=0.0 --clip_grad_norm=1.0 --d_obs_uni=128 --d_act_uni=21 --n_layers=4 --n_heads=8 --d_embd=256 --ctx_len=256 --seq_len=256 --mask_type="causal" --env_id=None --n_iters_rollout=100 --percent_data=0.25
python icl_bc_ed.py --seed=0 --load_ckpt=None --save_dir="/data/vision/phillipi/akumar01/synthetic-mdps-data/train_bc//mujoco/all-Ant"                    --n_ckpts=1 --obj="bc" --dataset_paths="/data/vision/phillipi/akumar01/synthetic-mdps-data/datasets/mujoco/*/dataset.pkl" --exclude_dataset_paths="/data/vision/phillipi/akumar01/synthetic-mdps-data/datasets/mujoco/Ant/dataset.pkl"                    --n_augs=0 --aug_dist="uniform" --nv=4096 --nh=131072 --n_iters_eval=10 --n_iters=50000 --bs=64 --mini_bs=64 --lr=0.0003 --lr_schedule="constant" --weight_decay=0.0 --clip_grad_norm=1.0 --d_obs_uni=128 --d_act_uni=21 --n_layers=4 --n_heads=8 --d_embd=256 --ctx_len=256 --seq_len=256 --mask_type="causal" --env_id=None --n_iters_rollout=100 --percent_data=0.25
python icl_bc_ed.py --seed=0 --load_ckpt=None --save_dir="/data/vision/phillipi/akumar01/synthetic-mdps-data/train_bc//mujoco/all-Humanoid"               --n_ckpts=1 --obj="bc" --dataset_paths="/data/vision/phillipi/akumar01/synthetic-mdps-data/datasets/mujoco/*/dataset.pkl" --exclude_dataset_paths="/data/vision/phillipi/akumar01/synthetic-mdps-data/datasets/mujoco/Humanoid/dataset.pkl"               --n_augs=0 --aug_dist="uniform" --nv=4096 --nh=131072 --n_iters_eval=10 --n_iters=50000 --bs=64 --mini_bs=64 --lr=0.0003 --lr_schedule="constant" --weight_decay=0.0 --clip_grad_norm=1.0 --d_obs_uni=128 --d_act_uni=21 --n_layers=4 --n_heads=8 --d_embd=256 --ctx_len=256 --seq_len=256 --mask_type="causal" --env_id=None --n_iters_rollout=100 --percent_data=0.25
python icl_bc_ed.py --seed=0 --load_ckpt=None --save_dir="/data/vision/phillipi/akumar01/synthetic-mdps-data/train_bc//mujoco/all-HumanoidStandup"        --n_ckpts=1 --obj="bc" --dataset_paths="/data/vision/phillipi/akumar01/synthetic-mdps-data/datasets/mujoco/*/dataset.pkl" --exclude_dataset_paths="/data/vision/phillipi/akumar01/synthetic-mdps-data/datasets/mujoco/HumanoidStandup/dataset.pkl"        --n_augs=0 --aug_dist="uniform" --nv=4096 --nh=131072 --n_iters_eval=10 --n_iters=50000 --bs=64 --mini_bs=64 --lr=0.0003 --lr_schedule="constant" --weight_decay=0.0 --clip_grad_norm=1.0 --d_obs_uni=128 --d_act_uni=21 --n_layers=4 --n_heads=8 --d_embd=256 --ctx_len=256 --seq_len=256 --mask_type="causal" --env_id=None --n_iters_rollout=100 --percent_data=0.25