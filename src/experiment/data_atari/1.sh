#!/bin/bash
source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps-atari/bin/activate
cd /data/vision/phillipi/akumar01/synthetic-mdps/src



CUDA_VISIBLE_DEVICES=0 python ppo_atari_envpool.py --track=True --env_id="Amidar-v5"        --save_dir="../data/exp_icl//datasets//real/atari/Amidar//"        &
CUDA_VISIBLE_DEVICES=1 python ppo_atari_envpool.py --track=True --env_id="Freeway-v5"       --save_dir="../data/exp_icl//datasets//real/atari/Freeway//"       &
CUDA_VISIBLE_DEVICES=2 python ppo_atari_envpool.py --track=True --env_id="Boxing-v5"        --save_dir="../data/exp_icl//datasets//real/atari/Boxing//"        &
CUDA_VISIBLE_DEVICES=3 python ppo_atari_envpool.py --track=True --env_id="Jamesbond-v5"     --save_dir="../data/exp_icl//datasets//real/atari/Jamesbond//"     &
wait
