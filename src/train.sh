

#python train.py --n_seeds=6 --agent=linear_transformer --env="env=dsmdp;n_states=64;d_obs=64;rpo=F;tl=4"   --save_agent="./data/agent_004.pkl" --n_envs=64 --n_steps=128 --total_timesteps=100e6
#python train.py --n_seeds=6 --agent=linear_transformer --env="env=dsmdp;n_states=64;d_obs=64;rpo=F;tl=16"  --save_agent="./data/agent_016.pkl" --n_envs=64 --n_steps=128 --total_timesteps=100e6
#python train.py --n_seeds=6 --agent=linear_transformer --env="env=dsmdp;n_states=64;d_obs=64;rpo=F;tl=32"  --save_agent="./data/agent_032.pkl" --n_envs=64 --n_steps=128 --total_timesteps=100e6
#python train.py --n_seeds=6 --agent=linear_transformer --env="env=dsmdp;n_states=64;d_obs=64;rpo=F;tl=128"  --save_agent="./data/agent_032.pkl" --n_envs=64 --n_steps=128 --total_timesteps=100e6



#CUDA_VISIBLE_DEVICES=0 python train.py --n_seeds=8 --agent=linear_transformer --env="env=dsmdp;n_states=64;n_acts=4;d_obs=64;rpo=64;tl=4"             --save_agent="./data/agent_0.pkl" --n_envs=64 --n_steps=128 --total_timesteps=100e6 &
#CUDA_VISIBLE_DEVICES=1 python train.py --n_seeds=8 --agent=linear_transformer --env="env=dsmdp;n_states=64;n_acts=4;d_obs=64;rpo=64;tl=128"           --save_agent="./data/agent_1.pkl" --n_envs=64 --n_steps=128 --total_timesteps=100e6 &
#CUDA_VISIBLE_DEVICES=2 python train.py --n_seeds=8 --agent=linear_transformer --env="env=csmdp;d_state=8;n_acts=4;d_obs=64;delta=F;rpo=64;tl=128"     --save_agent="./data/agent_2.pkl" --n_envs=64 --n_steps=128 --total_timesteps=100e6 &
#CUDA_VISIBLE_DEVICES=3 python train.py --n_seeds=8 --agent=linear_transformer --env="env=csmdp;d_state=8;n_acts=4;d_obs=64;delta=T;rpo=64;tl=128"     --save_agent="./data/agent_3.pkl" --n_envs=64 --n_steps=128 --total_timesteps=100e6 &
#wait



#python train.py --n_seeds=6 --agent=linear_transformer --env="env=dsmdp;n_states=64;n_acts=4;d_obs=64;rpo=64;tl=128" --save_agent="./data/rpo64.pkl" --n_envs=64 --n_steps=128 --total_timesteps=100e6


#python train.py --n_seeds=6 --agent=linear_transformer --env="env=dsmdp;n_states=64;n_acts=3;d_obs=64;rpo=64;tl=128" --save_agent="./data/temp.pkl" --n_envs=64 --n_steps=128 --total_timesteps=100e6



CUDA_VISIBLE_DEVICES=0 python train.py --n_seeds=8 --agent=linear_transformer --env="env=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=4" --save_dir="../data/env=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=4" --n_envs=64 --n_steps=128 --total_timesteps=10e6 &
CUDA_VISIBLE_DEVICES=1 python train.py --n_seeds=8 --agent=linear_transformer --env="env=dsmdp;n_states=64;n_acts=3;d_obs=64;rdist=U;rpo=64;tl=4" --save_dir="../data/env=dsmdp;n_states=64;n_acts=3;d_obs=64;rdist=U;rpo=64;tl=4" --n_envs=64 --n_steps=128 --total_timesteps=10e6 &
wait


python train.py --n_seeds=4 --agent=linear_transformer --env="env=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=4" --save_dir="../data/env=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=4" --n_envs=64 --n_steps=128 --total_timesteps=100e6

python train.py --n_seeds=4 --agent=linear_transformer --env="env=gridenv;grid_len=8;fobs=T;rpo=64;tl=128" --save_dir="../data/env=gridenv;grid_len=8;fobs=T;rpo=64;tl=128" --load_dir="../data/env=dsmdp;n_states=64;n_acts=4;d_obs=64;rdist=N;rpo=64;tl=4" --train=False --n_envs=64 --n_steps=128 --total_timesteps=3e6


