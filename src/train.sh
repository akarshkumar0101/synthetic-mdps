

#python train.py --n_seeds=6 --agent=linear_transformer --env="env=dsmdp;n_states=64;d_obs=64;rpo=F;tl=4"   --save_agent="./data/agent_004.pkl" --n_envs=64 --n_steps=128 --total_timesteps=100e6
#python train.py --n_seeds=6 --agent=linear_transformer --env="env=dsmdp;n_states=64;d_obs=64;rpo=F;tl=16"  --save_agent="./data/agent_016.pkl" --n_envs=64 --n_steps=128 --total_timesteps=100e6
#python train.py --n_seeds=6 --agent=linear_transformer --env="env=dsmdp;n_states=64;d_obs=64;rpo=F;tl=32"  --save_agent="./data/agent_032.pkl" --n_envs=64 --n_steps=128 --total_timesteps=100e6
python train.py --n_seeds=6 --agent=linear_transformer --env="env=dsmdp;n_states=64;d_obs=64;rpo=F;tl=128" --save_agent="./data/agent_128.pkl" --n_envs=64 --n_steps=128 --total_timesteps=100e6
python train.py --n_seeds=6 --agent=linear_transformer --env="env=dsmdp;n_states=64;d_obs=64;rpo=T;tl=128" --save_agent="./data/agent_128.pkl" --n_envs=64 --n_steps=128 --total_timesteps=100e6
