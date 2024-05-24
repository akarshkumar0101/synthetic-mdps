python icl_train.py --seed=0 --load_ckpt=None --save_dir="/vision-nfs/isola/env/akumar01/synthetic-mdps-data/icl_train/Reacher_oracle"                    --save_ckpt=True --dataset_paths="/vision-nfs/isola/env/akumar01/synthetic-mdps-data/datasets/mujoco/Reacher/dataset.pkl"                --exclude_dataset_paths=None --n_augs=0       --n_iters=20000  --env_id="Reacher"                --n_envs=64
python icl_train.py --seed=0 --load_ckpt=None --save_dir="/vision-nfs/isola/env/akumar01/synthetic-mdps-data/icl_train/Reacher_oracle_aug"                --save_ckpt=True --dataset_paths="/vision-nfs/isola/env/akumar01/synthetic-mdps-data/datasets/mujoco/Reacher/dataset.pkl"                --exclude_dataset_paths=None --n_augs=1000000 --n_iters=100000 --env_id="Reacher"                --n_envs=64
python icl_train.py --seed=0 --load_ckpt=None --save_dir="/vision-nfs/isola/env/akumar01/synthetic-mdps-data/icl_train/InvertedPendulum_oracle"           --save_ckpt=True --dataset_paths="/vision-nfs/isola/env/akumar01/synthetic-mdps-data/datasets/mujoco/InvertedPendulum/dataset.pkl"       --exclude_dataset_paths=None --n_augs=0       --n_iters=20000  --env_id="InvertedPendulum"       --n_envs=64
python icl_train.py --seed=0 --load_ckpt=None --save_dir="/vision-nfs/isola/env/akumar01/synthetic-mdps-data/icl_train/InvertedPendulum_oracle_aug"       --save_ckpt=True --dataset_paths="/vision-nfs/isola/env/akumar01/synthetic-mdps-data/datasets/mujoco/InvertedPendulum/dataset.pkl"       --exclude_dataset_paths=None --n_augs=1000000 --n_iters=100000 --env_id="InvertedPendulum"       --n_envs=64
python icl_train.py --seed=0 --load_ckpt=None --save_dir="/vision-nfs/isola/env/akumar01/synthetic-mdps-data/icl_train/InvertedDoublePendulum_oracle"     --save_ckpt=True --dataset_paths="/vision-nfs/isola/env/akumar01/synthetic-mdps-data/datasets/mujoco/InvertedDoublePendulum/dataset.pkl" --exclude_dataset_paths=None --n_augs=0       --n_iters=20000  --env_id="InvertedDoublePendulum" --n_envs=64
python icl_train.py --seed=0 --load_ckpt=None --save_dir="/vision-nfs/isola/env/akumar01/synthetic-mdps-data/icl_train/InvertedDoublePendulum_oracle_aug" --save_ckpt=True --dataset_paths="/vision-nfs/isola/env/akumar01/synthetic-mdps-data/datasets/mujoco/InvertedDoublePendulum/dataset.pkl" --exclude_dataset_paths=None --n_augs=1000000 --n_iters=100000 --env_id="InvertedDoublePendulum" --n_envs=64
python icl_train.py --seed=0 --load_ckpt=None --save_dir="/vision-nfs/isola/env/akumar01/synthetic-mdps-data/icl_train/HalfCheetah_oracle"                --save_ckpt=True --dataset_paths="/vision-nfs/isola/env/akumar01/synthetic-mdps-data/datasets/mujoco/HalfCheetah/dataset.pkl"            --exclude_dataset_paths=None --n_augs=0       --n_iters=20000  --env_id="HalfCheetah"            --n_envs=64
python icl_train.py --seed=0 --load_ckpt=None --save_dir="/vision-nfs/isola/env/akumar01/synthetic-mdps-data/icl_train/HalfCheetah_oracle_aug"            --save_ckpt=True --dataset_paths="/vision-nfs/isola/env/akumar01/synthetic-mdps-data/datasets/mujoco/HalfCheetah/dataset.pkl"            --exclude_dataset_paths=None --n_augs=1000000 --n_iters=100000 --env_id="HalfCheetah"            --n_envs=64
python icl_train.py --seed=0 --load_ckpt=None --save_dir="/vision-nfs/isola/env/akumar01/synthetic-mdps-data/icl_train/Hopper_oracle"                     --save_ckpt=True --dataset_paths="/vision-nfs/isola/env/akumar01/synthetic-mdps-data/datasets/mujoco/Hopper/dataset.pkl"                 --exclude_dataset_paths=None --n_augs=0       --n_iters=20000  --env_id="Hopper"                 --n_envs=64
python icl_train.py --seed=0 --load_ckpt=None --save_dir="/vision-nfs/isola/env/akumar01/synthetic-mdps-data/icl_train/Hopper_oracle_aug"                 --save_ckpt=True --dataset_paths="/vision-nfs/isola/env/akumar01/synthetic-mdps-data/datasets/mujoco/Hopper/dataset.pkl"                 --exclude_dataset_paths=None --n_augs=1000000 --n_iters=100000 --env_id="Hopper"                 --n_envs=64
python icl_train.py --seed=0 --load_ckpt=None --save_dir="/vision-nfs/isola/env/akumar01/synthetic-mdps-data/icl_train/Walker2d_oracle"                   --save_ckpt=True --dataset_paths="/vision-nfs/isola/env/akumar01/synthetic-mdps-data/datasets/mujoco/Walker2d/dataset.pkl"               --exclude_dataset_paths=None --n_augs=0       --n_iters=20000  --env_id="Walker2d"               --n_envs=64
python icl_train.py --seed=0 --load_ckpt=None --save_dir="/vision-nfs/isola/env/akumar01/synthetic-mdps-data/icl_train/Walker2d_oracle_aug"               --save_ckpt=True --dataset_paths="/vision-nfs/isola/env/akumar01/synthetic-mdps-data/datasets/mujoco/Walker2d/dataset.pkl"               --exclude_dataset_paths=None --n_augs=1000000 --n_iters=100000 --env_id="Walker2d"               --n_envs=64
python icl_train.py --seed=0 --load_ckpt=None --save_dir="/vision-nfs/isola/env/akumar01/synthetic-mdps-data/icl_train/Ant_oracle"                        --save_ckpt=True --dataset_paths="/vision-nfs/isola/env/akumar01/synthetic-mdps-data/datasets/mujoco/Ant/dataset.pkl"                    --exclude_dataset_paths=None --n_augs=0       --n_iters=20000  --env_id="Ant"                    --n_envs=64
python icl_train.py --seed=0 --load_ckpt=None --save_dir="/vision-nfs/isola/env/akumar01/synthetic-mdps-data/icl_train/Ant_oracle_aug"                    --save_ckpt=True --dataset_paths="/vision-nfs/isola/env/akumar01/synthetic-mdps-data/datasets/mujoco/Ant/dataset.pkl"                    --exclude_dataset_paths=None --n_augs=1000000 --n_iters=100000 --env_id="Ant"                    --n_envs=64