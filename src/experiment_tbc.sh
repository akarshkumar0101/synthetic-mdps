#!/bin/bash
source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps/bin/activate
cd /data/vision/phillipi/akumar01/synthetic-mdps/src




python viz_util.py --env_id="name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=linear;rew=linear;mrl=4x64" --save_dir="../data/exp_tbc//viz/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=linear;rew=linear;mrl=4x64"
python viz_util.py --env_id="name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=linear;rew=goal;mrl=4x64"   --save_dir="../data/exp_tbc//viz/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=linear;rew=goal;mrl=4x64"  
python viz_util.py --env_id="name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=linear;mrl=4x64"    --save_dir="../data/exp_tbc//viz/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=linear;mrl=4x64"   
python viz_util.py --env_id="name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=goal;mrl=4x64"      --save_dir="../data/exp_tbc//viz/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=goal;mrl=4x64"     
python viz_util.py --env_id="name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=T;trans=linear;rew=linear;mrl=4x64" --save_dir="../data/exp_tbc//viz/name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=T;trans=linear;rew=linear;mrl=4x64"
python viz_util.py --env_id="name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=T;trans=linear;rew=goal;mrl=4x64"   --save_dir="../data/exp_tbc//viz/name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=T;trans=linear;rew=goal;mrl=4x64"  

CUDA_VISIBLE_DEVICES=0 python run.py --env_id="name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=linear;rew=linear;mrl=4x64" --save_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=linear;rew=linear;mrl=4x64" --save_agent_params=True --n_iters=50 --n_envs=128 --n_envs_batch=32 &
CUDA_VISIBLE_DEVICES=1 python run.py --env_id="name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=linear;rew=goal;mrl=4x64"   --save_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=linear;rew=goal;mrl=4x64"   --save_agent_params=True --n_iters=50 --n_envs=128 --n_envs_batch=32 &
CUDA_VISIBLE_DEVICES=2 python run.py --env_id="name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=linear;mrl=4x64"    --save_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=linear;mrl=4x64"    --save_agent_params=True --n_iters=50 --n_envs=128 --n_envs_batch=32 &
CUDA_VISIBLE_DEVICES=3 python run.py --env_id="name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=goal;mrl=4x64"      --save_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=goal;mrl=4x64"      --save_agent_params=True --n_iters=50 --n_envs=128 --n_envs_batch=32 &
CUDA_VISIBLE_DEVICES=4 python run.py --env_id="name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=T;trans=linear;rew=linear;mrl=4x64" --save_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=T;trans=linear;rew=linear;mrl=4x64" --save_agent_params=True --n_iters=50 --n_envs=128 --n_envs_batch=32 &
CUDA_VISIBLE_DEVICES=5 python run.py --env_id="name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=T;trans=linear;rew=goal;mrl=4x64"   --save_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=T;trans=linear;rew=goal;mrl=4x64"   --save_agent_params=True --n_iters=50 --n_envs=128 --n_envs_batch=32 &
wait

CUDA_VISIBLE_DEVICES=0 python run.py --env_id="name=CartPole-v1;tl=500"           --save_dir="../data/exp_tbc//expert/name=CartPole-v1;tl=500"           --save_agent_params=True --n_iters=50 --n_envs=128 --n_envs_batch=32 &
CUDA_VISIBLE_DEVICES=1 python run.py --env_id="name=Acrobot-v1;tl=500"            --save_dir="../data/exp_tbc//expert/name=Acrobot-v1;tl=500"            --save_agent_params=True --n_iters=50 --n_envs=128 --n_envs_batch=32 &
CUDA_VISIBLE_DEVICES=2 python run.py --env_id="name=MountainCar-v0;tl=500"        --save_dir="../data/exp_tbc//expert/name=MountainCar-v0;tl=500"        --save_agent_params=True --n_iters=50 --n_envs=128 --n_envs_batch=32 &
CUDA_VISIBLE_DEVICES=3 python run.py --env_id="name=Asterix-MinAtar;tl=500"       --save_dir="../data/exp_tbc//expert/name=Asterix-MinAtar;tl=500"       --save_agent_params=True --n_iters=50 --n_envs=128 --n_envs_batch=32 &
CUDA_VISIBLE_DEVICES=4 python run.py --env_id="name=Breakout-MinAtar;tl=500"      --save_dir="../data/exp_tbc//expert/name=Breakout-MinAtar;tl=500"      --save_agent_params=True --n_iters=50 --n_envs=128 --n_envs_batch=32 &
CUDA_VISIBLE_DEVICES=5 python run.py --env_id="name=SpaceInvaders-MinAtar;tl=500" --save_dir="../data/exp_tbc//expert/name=SpaceInvaders-MinAtar;tl=500" --save_agent_params=True --n_iters=50 --n_envs=128 --n_envs_batch=32 &
wait

CUDA_VISIBLE_DEVICES=0 python run_bc.py --n_seeds=2 --env_id="name=CartPole-v1;tl=500"           --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=linear;rew=linear;mrl=4x64" --load_dir_teacher="../data/exp_tbc//expert/name=CartPole-v1;tl=500"           --save_dir="../data/exp_tbc//test/name=CartPole-v1;tl=500/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=linear;rew=linear;mrl=4x64"           --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=1 python run_bc.py --n_seeds=2 --env_id="name=CartPole-v1;tl=500"           --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=linear;rew=goal;mrl=4x64"   --load_dir_teacher="../data/exp_tbc//expert/name=CartPole-v1;tl=500"           --save_dir="../data/exp_tbc//test/name=CartPole-v1;tl=500/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=linear;rew=goal;mrl=4x64"             --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=2 python run_bc.py --n_seeds=2 --env_id="name=CartPole-v1;tl=500"           --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=linear;mrl=4x64"    --load_dir_teacher="../data/exp_tbc//expert/name=CartPole-v1;tl=500"           --save_dir="../data/exp_tbc//test/name=CartPole-v1;tl=500/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=linear;mrl=4x64"              --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=3 python run_bc.py --n_seeds=2 --env_id="name=CartPole-v1;tl=500"           --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=goal;mrl=4x64"      --load_dir_teacher="../data/exp_tbc//expert/name=CartPole-v1;tl=500"           --save_dir="../data/exp_tbc//test/name=CartPole-v1;tl=500/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=goal;mrl=4x64"                --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=4 python run_bc.py --n_seeds=2 --env_id="name=CartPole-v1;tl=500"           --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=T;trans=linear;rew=linear;mrl=4x64" --load_dir_teacher="../data/exp_tbc//expert/name=CartPole-v1;tl=500"           --save_dir="../data/exp_tbc//test/name=CartPole-v1;tl=500/name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=T;trans=linear;rew=linear;mrl=4x64"           --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=5 python run_bc.py --n_seeds=2 --env_id="name=CartPole-v1;tl=500"           --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=T;trans=linear;rew=goal;mrl=4x64"   --load_dir_teacher="../data/exp_tbc//expert/name=CartPole-v1;tl=500"           --save_dir="../data/exp_tbc//test/name=CartPole-v1;tl=500/name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=T;trans=linear;rew=goal;mrl=4x64"             --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
wait
CUDA_VISIBLE_DEVICES=0 python run_bc.py --n_seeds=2 --env_id="name=CartPole-v1;tl=500"           --run="eval"  --load_dir=None                                                                                                       --load_dir_teacher="../data/exp_tbc//expert/name=CartPole-v1;tl=500"           --save_dir="../data/exp_tbc//test/name=CartPole-v1;tl=500/random_agent"                                                                             --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=1 python run_bc.py --n_seeds=2 --env_id="name=CartPole-v1;tl=500"           --run="train" --load_dir=None                                                                                                       --load_dir_teacher="../data/exp_tbc//expert/name=CartPole-v1;tl=500"           --save_dir="../data/exp_tbc//test/name=CartPole-v1;tl=500/train_random"                                                                             --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=2 python run_bc.py --n_seeds=2 --env_id="name=Acrobot-v1;tl=500"            --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=linear;rew=linear;mrl=4x64" --load_dir_teacher="../data/exp_tbc//expert/name=Acrobot-v1;tl=500"            --save_dir="../data/exp_tbc//test/name=Acrobot-v1;tl=500/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=linear;rew=linear;mrl=4x64"            --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=3 python run_bc.py --n_seeds=2 --env_id="name=Acrobot-v1;tl=500"            --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=linear;rew=goal;mrl=4x64"   --load_dir_teacher="../data/exp_tbc//expert/name=Acrobot-v1;tl=500"            --save_dir="../data/exp_tbc//test/name=Acrobot-v1;tl=500/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=linear;rew=goal;mrl=4x64"              --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=4 python run_bc.py --n_seeds=2 --env_id="name=Acrobot-v1;tl=500"            --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=linear;mrl=4x64"    --load_dir_teacher="../data/exp_tbc//expert/name=Acrobot-v1;tl=500"            --save_dir="../data/exp_tbc//test/name=Acrobot-v1;tl=500/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=linear;mrl=4x64"               --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=5 python run_bc.py --n_seeds=2 --env_id="name=Acrobot-v1;tl=500"            --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=goal;mrl=4x64"      --load_dir_teacher="../data/exp_tbc//expert/name=Acrobot-v1;tl=500"            --save_dir="../data/exp_tbc//test/name=Acrobot-v1;tl=500/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=goal;mrl=4x64"                 --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
wait
CUDA_VISIBLE_DEVICES=0 python run_bc.py --n_seeds=2 --env_id="name=Acrobot-v1;tl=500"            --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=T;trans=linear;rew=linear;mrl=4x64" --load_dir_teacher="../data/exp_tbc//expert/name=Acrobot-v1;tl=500"            --save_dir="../data/exp_tbc//test/name=Acrobot-v1;tl=500/name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=T;trans=linear;rew=linear;mrl=4x64"            --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=1 python run_bc.py --n_seeds=2 --env_id="name=Acrobot-v1;tl=500"            --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=T;trans=linear;rew=goal;mrl=4x64"   --load_dir_teacher="../data/exp_tbc//expert/name=Acrobot-v1;tl=500"            --save_dir="../data/exp_tbc//test/name=Acrobot-v1;tl=500/name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=T;trans=linear;rew=goal;mrl=4x64"              --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=2 python run_bc.py --n_seeds=2 --env_id="name=Acrobot-v1;tl=500"            --run="eval"  --load_dir=None                                                                                                       --load_dir_teacher="../data/exp_tbc//expert/name=Acrobot-v1;tl=500"            --save_dir="../data/exp_tbc//test/name=Acrobot-v1;tl=500/random_agent"                                                                              --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=3 python run_bc.py --n_seeds=2 --env_id="name=Acrobot-v1;tl=500"            --run="train" --load_dir=None                                                                                                       --load_dir_teacher="../data/exp_tbc//expert/name=Acrobot-v1;tl=500"            --save_dir="../data/exp_tbc//test/name=Acrobot-v1;tl=500/train_random"                                                                              --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=4 python run_bc.py --n_seeds=2 --env_id="name=MountainCar-v0;tl=500"        --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=linear;rew=linear;mrl=4x64" --load_dir_teacher="../data/exp_tbc//expert/name=MountainCar-v0;tl=500"        --save_dir="../data/exp_tbc//test/name=MountainCar-v0;tl=500/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=linear;rew=linear;mrl=4x64"        --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=5 python run_bc.py --n_seeds=2 --env_id="name=MountainCar-v0;tl=500"        --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=linear;rew=goal;mrl=4x64"   --load_dir_teacher="../data/exp_tbc//expert/name=MountainCar-v0;tl=500"        --save_dir="../data/exp_tbc//test/name=MountainCar-v0;tl=500/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=linear;rew=goal;mrl=4x64"          --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
wait
CUDA_VISIBLE_DEVICES=0 python run_bc.py --n_seeds=2 --env_id="name=MountainCar-v0;tl=500"        --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=linear;mrl=4x64"    --load_dir_teacher="../data/exp_tbc//expert/name=MountainCar-v0;tl=500"        --save_dir="../data/exp_tbc//test/name=MountainCar-v0;tl=500/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=linear;mrl=4x64"           --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=1 python run_bc.py --n_seeds=2 --env_id="name=MountainCar-v0;tl=500"        --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=goal;mrl=4x64"      --load_dir_teacher="../data/exp_tbc//expert/name=MountainCar-v0;tl=500"        --save_dir="../data/exp_tbc//test/name=MountainCar-v0;tl=500/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=goal;mrl=4x64"             --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=2 python run_bc.py --n_seeds=2 --env_id="name=MountainCar-v0;tl=500"        --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=T;trans=linear;rew=linear;mrl=4x64" --load_dir_teacher="../data/exp_tbc//expert/name=MountainCar-v0;tl=500"        --save_dir="../data/exp_tbc//test/name=MountainCar-v0;tl=500/name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=T;trans=linear;rew=linear;mrl=4x64"        --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=3 python run_bc.py --n_seeds=2 --env_id="name=MountainCar-v0;tl=500"        --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=T;trans=linear;rew=goal;mrl=4x64"   --load_dir_teacher="../data/exp_tbc//expert/name=MountainCar-v0;tl=500"        --save_dir="../data/exp_tbc//test/name=MountainCar-v0;tl=500/name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=T;trans=linear;rew=goal;mrl=4x64"          --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=4 python run_bc.py --n_seeds=2 --env_id="name=MountainCar-v0;tl=500"        --run="eval"  --load_dir=None                                                                                                       --load_dir_teacher="../data/exp_tbc//expert/name=MountainCar-v0;tl=500"        --save_dir="../data/exp_tbc//test/name=MountainCar-v0;tl=500/random_agent"                                                                          --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=5 python run_bc.py --n_seeds=2 --env_id="name=MountainCar-v0;tl=500"        --run="train" --load_dir=None                                                                                                       --load_dir_teacher="../data/exp_tbc//expert/name=MountainCar-v0;tl=500"        --save_dir="../data/exp_tbc//test/name=MountainCar-v0;tl=500/train_random"                                                                          --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
wait
CUDA_VISIBLE_DEVICES=0 python run_bc.py --n_seeds=2 --env_id="name=Asterix-MinAtar;tl=500"       --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=linear;rew=linear;mrl=4x64" --load_dir_teacher="../data/exp_tbc//expert/name=Asterix-MinAtar;tl=500"       --save_dir="../data/exp_tbc//test/name=Asterix-MinAtar;tl=500/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=linear;rew=linear;mrl=4x64"       --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=1 python run_bc.py --n_seeds=2 --env_id="name=Asterix-MinAtar;tl=500"       --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=linear;rew=goal;mrl=4x64"   --load_dir_teacher="../data/exp_tbc//expert/name=Asterix-MinAtar;tl=500"       --save_dir="../data/exp_tbc//test/name=Asterix-MinAtar;tl=500/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=linear;rew=goal;mrl=4x64"         --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=2 python run_bc.py --n_seeds=2 --env_id="name=Asterix-MinAtar;tl=500"       --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=linear;mrl=4x64"    --load_dir_teacher="../data/exp_tbc//expert/name=Asterix-MinAtar;tl=500"       --save_dir="../data/exp_tbc//test/name=Asterix-MinAtar;tl=500/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=linear;mrl=4x64"          --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=3 python run_bc.py --n_seeds=2 --env_id="name=Asterix-MinAtar;tl=500"       --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=goal;mrl=4x64"      --load_dir_teacher="../data/exp_tbc//expert/name=Asterix-MinAtar;tl=500"       --save_dir="../data/exp_tbc//test/name=Asterix-MinAtar;tl=500/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=goal;mrl=4x64"            --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=4 python run_bc.py --n_seeds=2 --env_id="name=Asterix-MinAtar;tl=500"       --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=T;trans=linear;rew=linear;mrl=4x64" --load_dir_teacher="../data/exp_tbc//expert/name=Asterix-MinAtar;tl=500"       --save_dir="../data/exp_tbc//test/name=Asterix-MinAtar;tl=500/name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=T;trans=linear;rew=linear;mrl=4x64"       --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=5 python run_bc.py --n_seeds=2 --env_id="name=Asterix-MinAtar;tl=500"       --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=T;trans=linear;rew=goal;mrl=4x64"   --load_dir_teacher="../data/exp_tbc//expert/name=Asterix-MinAtar;tl=500"       --save_dir="../data/exp_tbc//test/name=Asterix-MinAtar;tl=500/name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=T;trans=linear;rew=goal;mrl=4x64"         --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
wait
CUDA_VISIBLE_DEVICES=0 python run_bc.py --n_seeds=2 --env_id="name=Asterix-MinAtar;tl=500"       --run="eval"  --load_dir=None                                                                                                       --load_dir_teacher="../data/exp_tbc//expert/name=Asterix-MinAtar;tl=500"       --save_dir="../data/exp_tbc//test/name=Asterix-MinAtar;tl=500/random_agent"                                                                         --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=1 python run_bc.py --n_seeds=2 --env_id="name=Asterix-MinAtar;tl=500"       --run="train" --load_dir=None                                                                                                       --load_dir_teacher="../data/exp_tbc//expert/name=Asterix-MinAtar;tl=500"       --save_dir="../data/exp_tbc//test/name=Asterix-MinAtar;tl=500/train_random"                                                                         --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=2 python run_bc.py --n_seeds=2 --env_id="name=Breakout-MinAtar;tl=500"      --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=linear;rew=linear;mrl=4x64" --load_dir_teacher="../data/exp_tbc//expert/name=Breakout-MinAtar;tl=500"      --save_dir="../data/exp_tbc//test/name=Breakout-MinAtar;tl=500/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=linear;rew=linear;mrl=4x64"      --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=3 python run_bc.py --n_seeds=2 --env_id="name=Breakout-MinAtar;tl=500"      --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=linear;rew=goal;mrl=4x64"   --load_dir_teacher="../data/exp_tbc//expert/name=Breakout-MinAtar;tl=500"      --save_dir="../data/exp_tbc//test/name=Breakout-MinAtar;tl=500/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=linear;rew=goal;mrl=4x64"        --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=4 python run_bc.py --n_seeds=2 --env_id="name=Breakout-MinAtar;tl=500"      --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=linear;mrl=4x64"    --load_dir_teacher="../data/exp_tbc//expert/name=Breakout-MinAtar;tl=500"      --save_dir="../data/exp_tbc//test/name=Breakout-MinAtar;tl=500/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=linear;mrl=4x64"         --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=5 python run_bc.py --n_seeds=2 --env_id="name=Breakout-MinAtar;tl=500"      --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=goal;mrl=4x64"      --load_dir_teacher="../data/exp_tbc//expert/name=Breakout-MinAtar;tl=500"      --save_dir="../data/exp_tbc//test/name=Breakout-MinAtar;tl=500/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=goal;mrl=4x64"           --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
wait
CUDA_VISIBLE_DEVICES=0 python run_bc.py --n_seeds=2 --env_id="name=Breakout-MinAtar;tl=500"      --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=T;trans=linear;rew=linear;mrl=4x64" --load_dir_teacher="../data/exp_tbc//expert/name=Breakout-MinAtar;tl=500"      --save_dir="../data/exp_tbc//test/name=Breakout-MinAtar;tl=500/name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=T;trans=linear;rew=linear;mrl=4x64"      --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=1 python run_bc.py --n_seeds=2 --env_id="name=Breakout-MinAtar;tl=500"      --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=T;trans=linear;rew=goal;mrl=4x64"   --load_dir_teacher="../data/exp_tbc//expert/name=Breakout-MinAtar;tl=500"      --save_dir="../data/exp_tbc//test/name=Breakout-MinAtar;tl=500/name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=T;trans=linear;rew=goal;mrl=4x64"        --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=2 python run_bc.py --n_seeds=2 --env_id="name=Breakout-MinAtar;tl=500"      --run="eval"  --load_dir=None                                                                                                       --load_dir_teacher="../data/exp_tbc//expert/name=Breakout-MinAtar;tl=500"      --save_dir="../data/exp_tbc//test/name=Breakout-MinAtar;tl=500/random_agent"                                                                        --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=3 python run_bc.py --n_seeds=2 --env_id="name=Breakout-MinAtar;tl=500"      --run="train" --load_dir=None                                                                                                       --load_dir_teacher="../data/exp_tbc//expert/name=Breakout-MinAtar;tl=500"      --save_dir="../data/exp_tbc//test/name=Breakout-MinAtar;tl=500/train_random"                                                                        --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=4 python run_bc.py --n_seeds=2 --env_id="name=SpaceInvaders-MinAtar;tl=500" --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=linear;rew=linear;mrl=4x64" --load_dir_teacher="../data/exp_tbc//expert/name=SpaceInvaders-MinAtar;tl=500" --save_dir="../data/exp_tbc//test/name=SpaceInvaders-MinAtar;tl=500/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=linear;rew=linear;mrl=4x64" --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=5 python run_bc.py --n_seeds=2 --env_id="name=SpaceInvaders-MinAtar;tl=500" --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=linear;rew=goal;mrl=4x64"   --load_dir_teacher="../data/exp_tbc//expert/name=SpaceInvaders-MinAtar;tl=500" --save_dir="../data/exp_tbc//test/name=SpaceInvaders-MinAtar;tl=500/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=linear;rew=goal;mrl=4x64"   --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
wait
CUDA_VISIBLE_DEVICES=0 python run_bc.py --n_seeds=2 --env_id="name=SpaceInvaders-MinAtar;tl=500" --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=linear;mrl=4x64"    --load_dir_teacher="../data/exp_tbc//expert/name=SpaceInvaders-MinAtar;tl=500" --save_dir="../data/exp_tbc//test/name=SpaceInvaders-MinAtar;tl=500/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=linear;mrl=4x64"    --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=1 python run_bc.py --n_seeds=2 --env_id="name=SpaceInvaders-MinAtar;tl=500" --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=goal;mrl=4x64"      --load_dir_teacher="../data/exp_tbc//expert/name=SpaceInvaders-MinAtar;tl=500" --save_dir="../data/exp_tbc//test/name=SpaceInvaders-MinAtar;tl=500/name=csmdp;d_state=2;d_obs=8;n_acts=4;delta=T;trans=mlp;rew=goal;mrl=4x64"      --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=2 python run_bc.py --n_seeds=2 --env_id="name=SpaceInvaders-MinAtar;tl=500" --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=T;trans=linear;rew=linear;mrl=4x64" --load_dir_teacher="../data/exp_tbc//expert/name=SpaceInvaders-MinAtar;tl=500" --save_dir="../data/exp_tbc//test/name=SpaceInvaders-MinAtar;tl=500/name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=T;trans=linear;rew=linear;mrl=4x64" --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=3 python run_bc.py --n_seeds=2 --env_id="name=SpaceInvaders-MinAtar;tl=500" --run="train" --load_dir="../data/exp_tbc//pretrain/name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=T;trans=linear;rew=goal;mrl=4x64"   --load_dir_teacher="../data/exp_tbc//expert/name=SpaceInvaders-MinAtar;tl=500" --save_dir="../data/exp_tbc//test/name=SpaceInvaders-MinAtar;tl=500/name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=T;trans=linear;rew=goal;mrl=4x64"   --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=4 python run_bc.py --n_seeds=2 --env_id="name=SpaceInvaders-MinAtar;tl=500" --run="eval"  --load_dir=None                                                                                                       --load_dir_teacher="../data/exp_tbc//expert/name=SpaceInvaders-MinAtar;tl=500" --save_dir="../data/exp_tbc//test/name=SpaceInvaders-MinAtar;tl=500/random_agent"                                                                   --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
CUDA_VISIBLE_DEVICES=5 python run_bc.py --n_seeds=2 --env_id="name=SpaceInvaders-MinAtar;tl=500" --run="train" --load_dir=None                                                                                                       --load_dir_teacher="../data/exp_tbc//expert/name=SpaceInvaders-MinAtar;tl=500" --save_dir="../data/exp_tbc//test/name=SpaceInvaders-MinAtar;tl=500/train_random"                                                                   --save_agent_params=True --n_iters=50 --n_envs_batch=4 &
wait